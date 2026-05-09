from __future__ import annotations

import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from shared.errors import exc_trace
from shared.litellm_usage import usage_snapshot_sync as usage_snapshot, diff as usage_diff

logger = logging.getLogger(__name__)


def _parse_locomo_date(date_str: str) -> datetime | None:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _date_epoch(date_str: str) -> int | None:
    parsed = _parse_locomo_date(date_str)
    return int(parsed.replace(tzinfo=timezone.utc).timestamp()) if parsed else None


def _sorted_sessions(sessions: Dict[str, Any], session_datetimes: Dict[str, Any]) -> list[tuple[str, str, list[dict]]]:
    keys = [k for k in sessions if re.match(r"^session_\d+$", k)]
    paired = [(k, session_datetimes.get(f"{k}_date_time", ""), sessions[k]) for k in keys]

    def sort_key(item: tuple) -> tuple:
        parsed = _parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def _turn_text(turn: dict) -> str:
    speaker = turn.get("speaker", "")
    text = turn.get("text", "")
    blip = turn.get("blip_caption") or turn.get("blip_captions") or ""
    query = turn.get("query", "")
    if query and blip:
        photo = f"[Sharing image - query: {query}. The image shows: {blip}]"
    elif query:
        photo = f"[Sharing image - query for: {query}]"
    elif blip:
        photo = f"[Sharing image that shows: {blip}]"
    else:
        photo = ""
    body = f"{text} {photo}".strip() if text else photo
    if not body:
        return ""
    return f"{speaker}: {body}" if speaker else body


class RagService:
    """Classical RAG baseline: embed each turn → Qdrant → top-k.

    Distinct from Mem0Service (which performs LLM-driven memory extraction)
    and GraphitiService (which builds a temporal graph). Same Qdrant
    instance, different collection — one per (conv, run)."""

    def __init__(self):
        self.TOP_K = int(os.getenv("MEMORIES_SEARCH_LIMIT", "20"))
        self.EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
        self.EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1536"))

        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        self.openai = OpenAI()
        self.run_id = uuid.uuid4().hex[:8]
        self.current_collection: str | None = None

    def _ensure_collection(self, name: str) -> None:
        if self.qdrant.collection_exists(name):
            return
        self.qdrant.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=self.EMBED_DIM, distance=qmodels.Distance.COSINE),
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.openai.embeddings.create(
            model=self.EMBED_MODEL,
            input=texts,
            dimensions=self.EMBED_DIM,
        )
        return [d.embedding for d in resp.data]

    def memorize_iter(self, conv_index: int, data: Dict[str, Any]):
        """Streaming generator: yields one event per turn, then done."""
        speaker_a = data.get("speaker_a", "")
        speaker_b = data.get("speaker_b", "")
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            yield {"event": "done", "status": "error", "reason": "'sessions' must be a dict",
                   "added": 0, "failed": 0}
            return

        collection = f"rag_locomo_{conv_index}_{self.run_id}"
        self.current_collection = collection
        self._ensure_collection(collection)

        sorted_sessions = _sorted_sessions(sessions, session_datetimes)
        logger.info(
            "RAG ingest: conv=%d %s & %s sessions=%d collection=%s",
            conv_index, speaker_a, speaker_b, len(sorted_sessions), collection,
        )

        added = 0
        failed = 0
        failures: List[Dict[str, Any]] = []

        for session_key, date_str, turns in sorted_sessions:
            session_epoch = _date_epoch(date_str)
            # Prepend the LOCOMO session timestamp in ISO 8601 UTC so the
            # stored chunk carries an absolute reference time inline —
            # same convention cognee/mem0 use, kept symmetric across
            # backends. RAG has no extractor that consumes it, so this
            # only affects the embedding vector and what the responder
            # reads back as context; the `payload.timestamp` field is
            # still written for any future filter logic that wants it.
            parsed = _parse_locomo_date(date_str)
            iso_prefix = parsed.replace(tzinfo=timezone.utc).isoformat() if parsed else None
            texts = [_turn_text(t) for t in turns]
            texts = [t for t in texts if t]
            if iso_prefix:
                texts = [f"[{iso_prefix}] {t}" for t in texts]
            if not texts:
                continue

            # `chunk_idx` is the wire-format key shared with mem0/graphiti/
            # full_context. Semantically it's the turn index within the session.
            for turn_idx, text in enumerate(texts):
                preview = text[:120].replace("\n", " ")
                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                try:
                    vec = self._embed([text])[0]
                    self.qdrant.upsert(
                        collection_name=collection,
                        points=[qmodels.PointStruct(
                            id=uuid.uuid4().hex,
                            vector=vec,
                            payload={
                                "text": text,
                                "session": session_key,
                                "timestamp": session_epoch,
                                "date": date_str,
                            },
                        )],
                        wait=True,
                    )
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    added += 1
                    logger.info("[rag load] conv=%d %s turn=%d wall=%.2fs edge=%d/$%.4f cloud=%d/$%.4f | %s",
                                conv_index, session_key, turn_idx, m_wall_ms / 1000,
                                du["edge_tokens"], du["edge_cost"], du["cloud_tokens"], du["cloud_cost"], preview)
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": turn_idx,
                        "status": "ok", "preview": preview, "error": None,
                        "wall_ms": m_wall_ms,
                        **du,
                    }
                except Exception as exc:
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    logger.exception("RAG ingest failed: conv %d %s turn %d",
                                     conv_index, session_key, turn_idx)
                    failed += 1
                    err_compact = exc_trace(exc)
                    failures.append({"session": session_key, "chunk_idx": turn_idx, "error": err_compact})
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": turn_idx,
                        "status": "failed", "preview": preview,
                        "error": err_compact[:300],
                        "wall_ms": m_wall_ms,
                        **du,
                    }

        yield {
            "event": "done",
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "collection": collection,
            "added": added,
            "failed": failed,
            "failures": failures or None,
        }

    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        memories: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        for evt in self.memorize_iter(conv_index, data):
            if evt.get("event") == "memory":
                memories.append({k: v for k, v in evt.items() if k != "event"})
            elif evt.get("event") == "done":
                summary = {k: v for k, v in evt.items() if k != "event"}
        summary["memories"] = memories
        return summary

    def warmup(self, conv_index: int) -> Dict[str, Any]:
        """Pre-create the per-conv Qdrant collection so its setup cost
        doesn't get folded into row #1 of the load loop."""
        collection = f"rag_locomo_{conv_index}_{self.run_id}"
        try:
            self._ensure_collection(collection)
            self.current_collection = collection
        except Exception:
            logger.exception("rag warmup ensure_collection failed")
        return {"backend": "rag", "warmed": True, "collection": collection}

    def reset(self) -> Dict[str, Any]:
        """Drop every rag_locomo_* collection in qdrant."""
        deleted = 0
        try:
            cols = self.qdrant.get_collections().collections
            for c in cols:
                if c.name.startswith("rag_locomo_"):
                    self.qdrant.delete_collection(c.name)
                    deleted += 1
        except Exception:
            logger.exception("rag reset: qdrant cleanup failed")
        self.run_id = uuid.uuid4().hex[:8]
        self.current_collection = None
        return {"backend": "rag", "deleted": deleted}

    def remember(self, question: str) -> List[str]:
        if not self.current_collection:
            logger.warning("No active collection — call memorize first.")
            return []

        try:
            qvec = self._embed([question])[0]
        except Exception:
            logger.exception("RAG query embedding failed")
            return []

        try:
            resp = self.qdrant.query_points(
                collection_name=self.current_collection,
                query=qvec,
                limit=self.TOP_K,
                with_payload=True,
            )
        except Exception:
            logger.exception("RAG search failed")
            return []

        results: List[str] = []
        for h in resp.points:
            payload = h.payload or {}
            text = (payload.get("text") or "").strip()
            if text:
                results.append(text)
        logger.info("RAG returned %d turns", len(results))
        return results
