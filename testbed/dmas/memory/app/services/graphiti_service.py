from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import timezone
from typing import Any, Dict, List

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)

from shared.errors import exc_trace
from shared.litellm_usage import usage_snapshot_sync as usage_snapshot, diff as usage_diff
from shared.openai_warmup import warmup_responses_async
from app.services.mem0_service import parse_locomo_date

logger = logging.getLogger(__name__)

MAX_SESSION_COUNT = 35

# Verbatim from getzep/zep-papers — kg_architecture_agent_memory/locomo_eval/
# zep_locomo_search.py. The preamble is what teaches the responder LLM how
# to read `event_time` annotations on facts; without it the bi-temporal
# data Graphiti stores is invisible to downstream reasoning.
ZEP_CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""


class GraphitiService:
    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

        if not (neo4j_uri and neo4j_user and neo4j_password):
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set")

        self.TOP_K = int(os.getenv("MEMORIES_SEARCH_LIMIT", "20"))

        # graphiti-core 0.29 defaults LLMConfig.temperature to 1.0 for
        # extraction; pin it to 0 so entity/relation extraction is
        # deterministic across benchmark runs. The cross-encoder
        # reranker is also pointed at gpt-4o-mini so the whole stack
        # uses a single cloud chat model (the SLM in ollama is the only
        # other model anywhere in the system).
        #
        # `small_model` MUST be set explicitly: graphiti's openai client
        # falls back to `DEFAULT_SMALL_MODEL = 'gpt-4.1-nano'` for any
        # task it decides is "small" (e.g. dedupe, classification).
        # LiteLLM only registers `gpt-4o-mini`, so without this pin the
        # small-model path 400s with `Invalid model name`. Pinning it to
        # the same `LLM_MODEL` keeps the small/big paths on one alias.
        cloud_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_config = LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=cloud_model,
            small_model=cloud_model,
            temperature=0,
        )
        llm_client = OpenAIClient(config=llm_config)
        cross_encoder = OpenAIRerankerClient(config=llm_config)
        self.graphiti = Graphiti(
            neo4j_uri, neo4j_user, neo4j_password,
            llm_client=llm_client,
            cross_encoder=cross_encoder,
        )
        self._initialized = False
        self.current_group_id: str | None = None

    async def _initialize(self):
        if not self._initialized:
            try:
                await self.graphiti.build_indices_and_constraints()
                self._initialized = True
                logger.info("Graphiti indices initialized")
            except Exception as e:
                logger.exception("Failed to initialize Graphiti indices: %s", e)

    async def warmup_async(self, conv_index: int) -> Dict[str, Any]:
        """Pay graphiti's one-time setup costs up front: Neo4j index +
        constraint build, then a litellm responses.parse ping so row #1
        of the load loop doesn't pay the cold OpenAI handshake."""
        await self._initialize()
        try:
            await self.graphiti.driver.execute_query(
                "CALL db.awaitIndexes($timeout)",
                timeout=600,
            )
        except Exception:
            logger.exception("warmup awaitIndexes failed")
        try:
            await warmup_responses_async()
        except Exception:
            logger.exception("graphiti warmup responses.parse failed")
        return {"backend": "graphiti", "warmed": True}

    async def memorize_iter_async(self, conv_index: int, data: Dict[str, Any]):
        """Async streaming generator: yields one event per add_episode then
        a final `{"event":"done"}` summary."""
        await self._initialize()

        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            yield {"event": "done", "status": "error", "reason": "'sessions' must be a dict",
                   "added": 0, "failed": 0}
            return

        group_id = f"locomo_experiment_user_{conv_index}"
        self.current_group_id = group_id

        total_msgs = sum(
            len(sessions.get(f"session_{i}") or [])
            for i in range(MAX_SESSION_COUNT)
            if session_datetimes.get(f"session_{i}_date_time") is not None
        )
        logger.info(
            "[graphiti load] conv=%d group_id=%s total_messages=%d",
            conv_index, group_id, total_msgs,
        )

        added = 0
        failed = 0
        failures: List[Dict[str, Any]] = []

        for session_idx in range(MAX_SESSION_COUNT):
            session_key = f"session_{session_idx}"
            session = sessions.get(session_key)
            if session is None:
                continue

            session_date_raw = session_datetimes.get(f"session_{session_idx}_date_time")
            if session_date_raw is None:
                continue

            # Reuse mem0's parser so graphiti accepts both `%B` (full) and
            # `%b` (abbreviated) month names — the prior strict `%B`-only
            # strptime silently dropped sessions with abbreviated names,
            # leaving them out of the graph while mem0/cognee ingested them.
            parsed = parse_locomo_date(session_date_raw)
            if parsed is None:
                logger.warning("Unparseable session date: %s", session_date_raw)
                continue
            date_string = parsed.replace(tzinfo=timezone.utc)

            for msg_idx, msg in enumerate(session):
                blip_caption = msg.get("blip_captions")
                img_description = (
                    f"(description of attached image: {blip_caption})"
                    if blip_caption is not None
                    else ""
                )

                episode_body = f"{msg.get('speaker', '')}: {msg.get('text', '')}{img_description}"
                episode_name = f"{group_id}_{session_key}_{added + failed}"
                idx = added + failed + 1
                preview = episode_body[:120].replace("\n", " ")

                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                try:
                    # Mirror zep_locomo_ingestion.py: data=body, type=message,
                    # created_at=session date, group_id=group_id. graphiti-core's
                    # `reference_time` is the datetime equivalent of Zep cloud's
                    # `created_at` ISO string. graphiti-core 0.29 made
                    # `source_description` a required positional arg; we pass an
                    # empty string to keep the entity-extraction prompt identical
                    # to Zep's harness, which provides no description.
                    await self.graphiti.add_episode(
                        name=episode_name,
                        episode_body=episode_body,
                        source=EpisodeType.message,
                        source_description="",
                        reference_time=date_string,
                        group_id=group_id,
                    )
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    added += 1
                    logger.info(
                        "[graphiti load] conv=%d %d/%d %s wall=%.2fs edge=%d/$%.4f cloud=%d/$%.4f | %s",
                        conv_index, idx, total_msgs, session_key, m_wall_ms / 1000,
                        du["edge_tokens"], du["edge_cost"], du["cloud_tokens"], du["cloud_cost"], preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": msg_idx,
                        "status": "ok", "preview": preview, "error": None,
                        "wall_ms": m_wall_ms,
                        **du,
                    }
                except Exception as exc:
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    logger.exception("Ingestion failed: conv %d %s", conv_index, session_key)
                    failed += 1
                    err_compact = exc_trace(exc)
                    failures.append({
                        "session": session_key,
                        "error": err_compact,
                    })
                    logger.info(
                        "[graphiti load] conv=%d %d/%d %s FAILED wall=%.2fs | %s",
                        conv_index, idx, total_msgs, session_key, m_wall_ms / 1000, preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": msg_idx,
                        "status": "failed", "preview": preview,
                        "error": err_compact[:300],
                        "wall_ms": m_wall_ms,
                        **du,
                    }

        # Block until Neo4j finishes async fulltext-index population so that
        # the caller's wall-clock measurement includes the real load cost.
        # graphiti-core has no warmup log to grep on; db.awaitIndexes is the
        # deterministic signal that the graph is fully queryable. The bench
        # now drives one message per call so this fires per message — fast
        # when no indexes are pending; demoted to debug to keep logs quiet.
        logger.debug("[graphiti load] conv=%d awaiting Neo4j indexes", conv_index)
        try:
            await self.graphiti.driver.execute_query(
                "CALL db.awaitIndexes($timeout)",
                timeout=600,
            )
        except Exception:
            logger.exception("db.awaitIndexes failed; load timing may be undercounted")

        yield {
            "event": "done",
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "group_id": group_id,
            "added": added,
            "failed": failed,
            "failures": failures if failures else None,
        }

    async def memorize_conversation_async(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        memories: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        async for evt in self.memorize_iter_async(conv_index, data):
            if evt.get("event") == "memory":
                memories.append({k: v for k, v in evt.items() if k != "event"})
            elif evt.get("event") == "done":
                summary = {k: v for k, v in evt.items() if k != "event"}
        summary["memories"] = memories
        return summary

    async def reset_async(self) -> Dict[str, Any]:
        """Wipe every API-reachable piece of Neo4j state without
        restarting the container: nodes + relationships + indexes +
        constraints, then clear the query plan cache and force a WAL
        checkpoint so transaction logs flush to store files. This is
        the deepest cleanup neo4j exposes via cypher; the only state
        that survives is the empty store-file allocations themselves
        (neo4j doesn't deallocate pages without restart, but the data
        in them is gone).

        The subsequent warmup re-runs `build_indices_and_constraints`
        to put the schema back."""
        try:
            await self.graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
        except Exception:
            logger.exception("graphiti reset: DETACH DELETE failed")
            return {"backend": "graphiti", "deleted": False}

        dropped_idx = dropped_con = 0
        try:
            rows, _, _ = await self.graphiti.driver.execute_query("SHOW INDEXES YIELD name")
            for r in rows:
                name = r["name"]
                try:
                    await self.graphiti.driver.execute_query(f"DROP INDEX `{name}` IF EXISTS")
                    dropped_idx += 1
                except Exception:
                    logger.exception("graphiti reset: DROP INDEX %s failed", name)
        except Exception:
            logger.exception("graphiti reset: SHOW INDEXES failed")
        try:
            rows, _, _ = await self.graphiti.driver.execute_query("SHOW CONSTRAINTS YIELD name")
            for r in rows:
                name = r["name"]
                try:
                    await self.graphiti.driver.execute_query(f"DROP CONSTRAINT `{name}` IF EXISTS")
                    dropped_con += 1
                except Exception:
                    logger.exception("graphiti reset: DROP CONSTRAINT %s failed", name)
        except Exception:
            logger.exception("graphiti reset: SHOW CONSTRAINTS failed")

        # Drop cached query plans and flush WAL -> store files. Closest
        # we can get to a volume wipe without restarting the container.
        try:
            await self.graphiti.driver.execute_query("CALL db.clearQueryCaches()")
        except Exception:
            logger.exception("graphiti reset: clearQueryCaches failed")
        try:
            await self.graphiti.driver.execute_query("CALL db.checkpoint()")
        except Exception:
            logger.exception("graphiti reset: checkpoint failed")

        # Force a re-init on next memorize so build_indices_and_constraints
        # runs again — the existing _initialized guard would otherwise
        # skip schema rebuild after we just dropped it.
        self._initialized = False
        self.current_group_id = None
        return {"backend": "graphiti", "deleted": True,
                "indexes_dropped": dropped_idx,
                "constraints_dropped": dropped_con}

    async def remember_async(self, question: str) -> List[str]:
        """Mirror Zep's zep_locomo_search.py verbatim:
          - edges: scope='edges', reranker='cross_encoder', limit=TOP_K
          - nodes: scope='nodes', reranker='rrf',           limit=TOP_K
        run in parallel, then merged into Zep's exact FACTS+ENTITIES
        context template. The single returned element IS that template;
        the route counts memories via len() on the returned list.
        """
        await self._initialize()

        if not self.current_group_id:
            logger.warning("No active group_id — call memorize first.")
            return []

        try:
            limit = self.TOP_K
            logger.info("Graphiti search: query=%r group_id=%r limit=%d (edges=%d, nodes=%d)",
                        question, self.current_group_id, limit, limit, limit)

            edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
            edge_config.limit = limit
            edges_task = self.graphiti._search(
                query=question,
                config=edge_config,
                group_ids=[self.current_group_id],
            )

            node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_config.limit = limit
            nodes_task = self.graphiti._search(
                query=question,
                config=node_config,
                group_ids=[self.current_group_id],
            )

            edge_results, node_results = await asyncio.gather(edges_task, nodes_task)

            edges = getattr(edge_results, "edges", []) if edge_results is not None else []
            entity_nodes = getattr(node_results, "nodes", []) if node_results is not None else []

            # Zep's compose_search_context: "  - {fact} (event_time: {valid_at})"
            # for facts; "  - {name}: {summary}" for entities.
            facts_lines = [
                f"  - {(getattr(e, 'fact', '') or '').strip()} (event_time: {getattr(e, 'valid_at', None)})"
                for e in edges or []
                if (getattr(e, 'fact', '') or '').strip()
            ]
            entities_lines = [
                f"  - {getattr(n, 'name', '')}: {(getattr(n, 'summary', '') or '').strip()}"
                for n in entity_nodes or []
                if (getattr(n, 'summary', '') or '').strip()
            ]
            block = ZEP_CONTEXT_TEMPLATE.format(
                facts='\n'.join(facts_lines),
                entities='\n'.join(entities_lines),
            )

            count = len(facts_lines) + len(entities_lines)
            logger.info("Graphiti returned %d facts + %d entity summaries",
                        len(facts_lines), len(entities_lines))
            return [block] if count else []
        except Exception:
            logger.exception("Graphiti search failed")
            return []
