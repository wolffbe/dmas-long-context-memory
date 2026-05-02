from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

# cognee 1.0 turned multi-user access control on by default, which
# requires the graph dataset-database handler to match the graph
# provider. Single-tenant benchmark — disable it before importing cognee
# (the value is read at import time). Same for session caching, which
# we don't want clouding the comparison.
os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")

import cognee
from cognee.api.v1.search import SearchType

# Side-effect import: registers the qdrant vector adapter with cognee.
# cognee 0.5.x ships only LanceDB/PGVector/Chroma in core; qdrant lives
# in the cognee-community adapter package and must be imported BEFORE
# any cognee call that resolves the vector backend, otherwise cognee
# raises "Unsupported vector database provider: qdrant".
import cognee_community_vector_adapter_qdrant.register  # noqa: F401

from app.services.litellm_usage import usage_snapshot, diff as usage_diff

logger = logging.getLogger(__name__)

MAX_SESSION_COUNT = 35


class CogneeService:
    """Cognee memory baseline: per-message `add` + `cognify` builds a
    knowledge graph in Neo4j with embeddings in Qdrant; `search` returns
    graph-completion context.

    Pinned to cognee 0.5.6 (V1 add/cognify/search API) so we can keep
    qdrant as the vector store via the cognee-community adapter — cognee
    1.0+ has the new remember/recall/forget/improve API, but no published
    qdrant adapter version is compatible with 1.0.x. Qdrant parity with
    rag/mem0 was the constraint that forced this version pin.

    Note on canonical-eval fidelity: unlike mem0 (mirrors
    mem0ai/memory-benchmarks) and graphiti (mirrors getzep/zep-papers
    locomo_eval), cognee's authors do NOT publish a LoCoMo harness —
    their eval framework targets HotpotQA / MuSiQue / 2WikiMultiHop
    (see evals/src/qa/qa_benchmark_cognee.py). Their HotpotQA flow is
    `prune_data + prune_system → add(corpus, batch) → run_pipeline once
    → search(SearchType.GRAPH_COMPLETION, top_k=5)`. We deliberately
    deviate by driving cognee per-message (one add+cognify per turn) to
    keep the bench's individual-ingestion contract identical across all
    five backends, even though it costs more LLM calls than cognee's
    own batched pattern. Search uses GRAPH_COMPLETION with TOP_K from
    MEMORIES_SEARCH_LIMIT (default 20) for parity with mem0/graphiti/rag.

    Shares Neo4j with GraphitiService and Qdrant with Mem0Service/RagService.
    The benchmark resets state between (backend, conv, mode) legs, so the
    services never run concurrently — but they must not run in parallel.
    """

    def __init__(self):
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        self.TOP_K = int(os.getenv("MEMORIES_SEARCH_LIMIT", "20"))

        # Match the rest of the stack on LLM + embedding choice — same
        # gpt-4o-mini and text-embedding-3-small@1536 routed through the
        # same litellm proxy that responder/judge/mem0/graphiti/rag use,
        # so extraction-time LLM and embedding cost are directly
        # comparable across backends. cognee 0.5.6 caches config at first
        # use, so set everything explicitly here rather than relying on
        # process env (which cognee may have already snapshotted).
        cognee.config.set_llm_provider("openai")
        cognee.config.set_llm_model(os.getenv("LLM_MODEL", "gpt-4o-mini"))
        cognee.config.set_llm_api_key(os.getenv("OPENAI_API_KEY", ""))
        # cognee 0.5.6 defaults llm_temperature to 0, but pin it
        # explicitly so it can't drift if a future config layer changes
        # the default. Mirrors the temperature=0 contract used at every
        # other LLM call site (responder, coordinator, judge, mem0,
        # graphiti).
        cognee.config.set_llm_config({"llm_temperature": 0})
        if os.getenv("OPENAI_BASE_URL"):
            cognee.config.set_llm_endpoint(os.environ["OPENAI_BASE_URL"])
        cognee.config.set_embedding_provider("openai")
        cognee.config.set_embedding_model(
            os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
        )
        cognee.config.set_embedding_dimensions(int(os.getenv("RAG_EMBED_DIM", "1536")))
        cognee.config.set_embedding_api_key(os.getenv("OPENAI_API_KEY", ""))
        if os.getenv("OPENAI_BASE_URL"):
            cognee.config.set_embedding_endpoint(os.environ["OPENAI_BASE_URL"])

        # Point cognee at the shared infra.
        cognee.config.set_graph_database_provider("neo4j")
        cognee.config.set_graph_db_config({
            "graph_database_url": neo4j_uri,
            "graph_database_username": neo4j_user,
            "graph_database_password": neo4j_password,
        })
        cognee.config.set_vector_db_config({
            "vector_db_provider": "qdrant",
            "vector_db_url": f"http://{qdrant_host}:{qdrant_port}",
            "vector_db_key": "",
        })
        # cognee's dataset-database handler is a separate routing layer
        # from the vector adapter; it also has to be told to use qdrant
        # so dataset metadata stays consistent with the vector store.
        os.environ.setdefault("VECTOR_DATASET_DATABASE_HANDLER", "qdrant")

        self.run_id = uuid.uuid4().hex[:8]
        self.current_dataset: str | None = None

    def _dataset_for(self, conv_index: int) -> str:
        return f"locomo_{conv_index}_{self.run_id}"

    async def _await_graph_indexes(self) -> None:
        """Block until Neo4j finishes async fulltext-index population.
        Mirrors graphiti's `db.awaitIndexes` pattern so per-ingest wall_ms
        captures the full cost, including index build."""
        try:
            from cognee.infrastructure.databases.graph import get_graph_engine
            engine = await get_graph_engine()
            driver = getattr(engine, "driver", None)
            if driver is not None:
                await driver.execute_query(
                    "CALL db.awaitIndexes($timeout)", timeout=600,
                )
        except Exception:
            logger.debug("cognee awaitIndexes skipped", exc_info=True)

    async def warmup_async(self, conv_index: int) -> Dict[str, Any]:
        """No-op cognify on an empty dataset to pay any one-time
        Neo4j-schema / Qdrant-collection setup cost up front."""
        dataset = self._dataset_for(conv_index)
        try:
            # Touch the dataset so cognee's relational metadata store
            # creates its rows; subsequent /memorize calls skip this.
            await cognee.add("warmup", dataset_name=dataset)
        except Exception:
            logger.exception("cognee warmup add failed")
        return {"backend": "cognee", "warmed": True, "dataset": dataset}

    def warmup(self, conv_index: int) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.warmup_async(conv_index))

    async def memorize_iter_async(self, conv_index: int, data: Dict[str, Any]):
        """Async streaming generator: per-message add+cognify, one event
        per message, then a final `{"event":"done"}` summary."""
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            yield {"event": "done", "status": "error", "reason": "'sessions' must be a dict",
                   "added": 0, "failed": 0}
            return

        dataset = self._dataset_for(conv_index)
        self.current_dataset = dataset

        total_msgs = sum(
            len(sessions.get(f"session_{i}") or [])
            for i in range(MAX_SESSION_COUNT)
            if session_datetimes.get(f"session_{i}_date_time") is not None
        )
        logger.info(
            "[cognee load] conv=%d dataset=%s total_messages=%d",
            conv_index, dataset, total_msgs,
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

            try:
                session_dt = datetime.strptime(
                    session_date_raw + " UTC", "%I:%M %p on %d %B, %Y UTC"
                ).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                logger.warning("Unparseable session date: %s", session_date_raw)
                continue

            for msg_idx, msg in enumerate(session):
                blip_caption = msg.get("blip_captions")
                img_description = (
                    f"(description of attached image: {blip_caption})"
                    if blip_caption is not None else ""
                )
                body = f"{msg.get('speaker','')}: {msg.get('text','')}{img_description}".strip()
                if not body or body == ":":
                    continue

                # Prepend the session timestamp so cognee's entity extractor
                # can pin facts in time the way graphiti's reference_time does.
                episode = f"[{session_dt.isoformat()}] {body}"
                idx = added + failed + 1
                preview = body[:120].replace("\n", " ")

                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                try:
                    await cognee.add(episode, dataset_name=dataset)
                    await cognee.cognify(datasets=[dataset])
                    # Flush: wait for Neo4j fulltext indexes to finish
                    # population before stamping wall_ms. Without this the
                    # measurement would undercount the real cost of this
                    # ingest (Neo4j builds indexes asynchronously).
                    await self._await_graph_indexes()
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    added += 1
                    logger.info(
                        "[cognee load] conv=%d %d/%d %s wall=%.2fs edge=%d/$%.4f cloud=%d/$%.4f | %s",
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
                    logger.exception("Ingestion failed: conv %d %s msg %d",
                                     conv_index, session_key, msg_idx)
                    failed += 1
                    failures.append({
                        "session": session_key,
                        "chunk_index": msg_idx,
                        "error": str(exc),
                    })
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": msg_idx,
                        "status": "failed", "preview": preview, "error": str(exc)[:300],
                        "wall_ms": m_wall_ms,
                        **du,
                    }

        yield {
            "event": "done",
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "dataset": dataset,
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

    # `memorize_iter` (sync) is needed by the streaming /memorize_stream
    # path. Drains the async iterator into a list and replays it; cognee
    # only exposes async APIs, so we can't yield each event the moment
    # the await resolves without an async route. The bench currently
    # uses the non-streaming /memorize endpoint via memorize_conversation.
    def memorize_iter(self, conv_index: int, data: Dict[str, Any]):
        async def _drain():
            return [e async for e in self.memorize_iter_async(conv_index, data)]
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        for evt in loop.run_until_complete(_drain()):
            yield evt

    async def reset_async(self) -> Dict[str, Any]:
        """Drop everything cognee has persisted: graph data, vectors,
        and its relational metadata. Note: this also wipes graphiti's
        graph since they share Neo4j — fine because the benchmark
        runs backends sequentially."""
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception:
            logger.exception("cognee reset: prune failed")
            return {"backend": "cognee", "deleted": False}
        self.current_dataset = None
        self.run_id = uuid.uuid4().hex[:8]
        return {"backend": "cognee", "deleted": True}

    def reset(self) -> Dict[str, Any]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.reset_async())

    async def remember_async(self, question: str) -> List[str]:
        if not self.current_dataset:
            logger.warning("No active dataset — call memorize first.")
            return []

        logger.info("Cognee search: query=%r dataset=%r limit=%d",
                    question, self.current_dataset, self.TOP_K)
        try:
            results = await cognee.search(
                query_text=question,
                query_type=SearchType.GRAPH_COMPLETION,
                datasets=[self.current_dataset],
                top_k=self.TOP_K,
            )
        except Exception:
            logger.exception("Cognee search failed")
            return []

        # cognee's search returns a list of strings or list of dicts
        # depending on query_type / version; normalize to strings and
        # cap at TOP_K so the comparison with other backends is fair.
        out: List[str] = []
        for item in (results or []):
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = (item.get("text") or item.get("content")
                        or item.get("answer") or "").strip()
            else:
                text = str(item).strip()
            if text:
                out.append(text)
            if len(out) >= self.TOP_K:
                break

        logger.info("Cognee returned %d items", len(out))
        return out

    def remember(self, question: str) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.remember_async(question))
