from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)

from app.services.litellm_usage import usage_snapshot, diff as usage_diff

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

        self.graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
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
        """Pay the one-time index/constraint build cost up front so it
        doesn't spike row #1 of the load loop."""
        await self._initialize()
        try:
            await self.graphiti.driver.execute_query(
                "CALL db.awaitIndexes($timeout)",
                timeout=600,
            )
        except Exception:
            logger.exception("warmup awaitIndexes failed")
        return {"backend": "graphiti", "warmed": True}

    def warmup(self, conv_index: int) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.warmup_async(conv_index))

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

            session_date = session_date_raw + " UTC"
            date_format = "%I:%M %p on %d %B, %Y UTC"
            try:
                date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                logger.warning("Unparseable session date: %s", session_date_raw)
                continue
            iso_date = date_string.isoformat()

            for msg_idx, msg in enumerate(session):
                blip_caption = msg.get("blip_captions")
                img_description = (
                    f"(description of attached image: {blip_caption})"
                    if blip_caption is not None
                    else ""
                )

                episode_body = msg.get("speaker") + ": " + msg.get("text") + img_description
                episode_name = f"{group_id}_{session_key}_{added + failed}"
                idx = added + failed + 1
                preview = episode_body[:120].replace("\n", " ")

                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                try:
                    # Mirror zep_locomo_ingestion.py: data=body, type=message,
                    # created_at=session date, group_id=group_id. graphiti-core's
                    # `reference_time` is the datetime equivalent of Zep cloud's
                    # `created_at` ISO string. No source_description — the Zep
                    # graph.add API has no equivalent, and passing one would
                    # leak extra signal into graphiti-core's entity-extraction
                    # prompt that Zep's harness never provides.
                    await self.graphiti.add_episode(
                        name=episode_name,
                        episode_body=episode_body,
                        source=EpisodeType.message,
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
                    failures.append({
                        "session": session_key,
                        "error": str(exc),
                    })
                    logger.info(
                        "[graphiti load] conv=%d %d/%d %s FAILED wall=%.2fs | %s",
                        conv_index, idx, total_msgs, session_key, m_wall_ms / 1000, preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": msg_idx,
                        "status": "failed", "preview": preview, "error": str(exc)[:300],
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

    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.memorize_conversation_async(conv_index, data))

    async def reset_async(self) -> Dict[str, Any]:
        """Wipe every node + relationship from Neo4j so the next /memorize
        starts on a clean graph. We drop the whole graph rather than
        per-group_id because graphiti's indexes are global."""
        try:
            await self.graphiti.driver.execute_query("MATCH (n) DETACH DELETE n")
        except Exception:
            logger.exception("graphiti reset: DETACH DELETE failed")
            return {"backend": "graphiti", "deleted": False}
        self.current_group_id = None
        # Indexes survive the data wipe; no need to rebuild.
        return {"backend": "graphiti", "deleted": True}

    def reset(self) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.reset_async())

    async def remember_async(self, question: str) -> List[str]:
        """Mirror Zep's zep_locomo_search.py verbatim:
          - edges: scope='edges', reranker='cross_encoder', limit=TOP_K
          - nodes: scope='nodes', reranker='rrf',           limit=TOP_K
        run in parallel, then merged into Zep's exact FACTS+ENTITIES
        context template. The single returned element IS that template
        — `count` is surfaced separately via `_last_search_count`.
        """
        await self._initialize()
        # Default to 0 so callers can still read it after a no-op return.
        self._last_search_count = 0

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

            self._last_search_count = len(facts_lines) + len(entities_lines)
            logger.info("Graphiti returned %d facts + %d entity summaries",
                        len(facts_lines), len(entities_lines))
            return [block] if self._last_search_count else []
        except Exception:
            logger.exception("Graphiti search failed")
            return []

    def remember(self, question: str) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.remember_async(question))
