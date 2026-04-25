from __future__ import annotations

import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from mem0 import Memory
from mem0.configs.base import MemoryConfig

from app.services.litellm_usage import usage_snapshot, diff as usage_diff

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1


def parse_locomo_date(date_str: str) -> datetime | None:
    """Parse LOCOMO date: '1:56 pm on 8 May, 2023'."""
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def locomo_date_to_epoch(date_str: str) -> int | None:
    parsed = parse_locomo_date(date_str)
    if parsed:
        return int(parsed.replace(tzinfo=timezone.utc).timestamp())
    return None


def get_sorted_sessions(sessions: Dict[str, Any], session_datetimes: Dict[str, Any]) -> list[tuple[str, str, list[dict]]]:
    """Extract and sort sessions chronologically."""
    session_keys = [k for k in sessions if re.match(r"^session_\d+$", k)]
    paired = []
    for key in session_keys:
        date_key = f"{key}_date_time"
        date_str = session_datetimes.get(date_key, "")
        turns = sessions[key]
        paired.append((key, date_str, turns))

    def sort_key(item: tuple) -> tuple:
        parsed = parse_locomo_date(item[1])
        if parsed:
            return (0, parsed)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, datetime(2000, 1, num))

    paired.sort(key=sort_key)
    return paired


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str) -> list[list[dict]]:
    """Convert turns to message chunks for ingestion."""
    messages = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo_tag = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo_tag = f"[Sharing image - query for: {query}]"
        elif blip:
            photo_tag = f"[Sharing image that shows: {blip}]"
        else:
            photo_tag = ""
        if photo_tag:
            text = f"{text} {photo_tag}" if text else photo_tag
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})

    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i : i + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
    return chunks


class Mem0Service:

    def __init__(self):
        self.TOP_K = int(os.getenv("MEMORIES_SEARCH_LIMIT", "20"))

        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                },
            }
        )
        self.memory = Memory(config)
        self.run_id = uuid.uuid4().hex[:8]
        self.current_user_id: str | None = None

    def memorize_iter(self, conv_index: int, data: Dict[str, Any]):
        """Streaming generator: yields one event per add then a final
        `{"event":"done"}` summary. Caller is responsible for consuming.
        """
        speaker_a = data.get("speaker_a", "")
        speaker_b = data.get("speaker_b", "")
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            yield {"event": "done", "status": "error", "reason": "'sessions' must be a dict",
                   "added": 0, "failed": 0}
            return

        user_id = f"locomo_{conv_index}_{self.run_id}"
        self.current_user_id = user_id

        sorted_sessions = get_sorted_sessions(sessions, session_datetimes)
        total_chunks = sum(len(session_to_chunks(s, speaker_a, speaker_b)) for _, _, s in sorted_sessions)

        logger.info(
            "[mem0 load] conv=%d %s & %s sessions=%d chunks=%d",
            conv_index, speaker_a, speaker_b, len(sorted_sessions), total_chunks,
        )

        added = 0
        failed = 0
        failures: List[Dict[str, Any]] = []

        for session_key, date_str, turns in sorted_sessions:
            chunks = session_to_chunks(turns, speaker_a, speaker_b)
            if not chunks:
                continue

            session_epoch = locomo_date_to_epoch(date_str)

            for chunk_idx, messages in enumerate(chunks):
                if any(not msg.get("content", "").strip() for msg in messages):
                    continue

                idx = added + failed + 1
                preview = (messages[0].get("content", "") if messages else "")[:120].replace("\n", " ")

                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                try:
                    self.memory.add(
                        messages,
                        user_id=user_id,
                        metadata={"timestamp": session_epoch},
                    )
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    added += 1
                    logger.info(
                        "[mem0 load] conv=%d %d/%d %s wall=%.2fs edge=%d/$%.4f cloud=%d/$%.4f | %s",
                        conv_index, idx, total_chunks, session_key, m_wall_ms / 1000,
                        du["edge_tokens"], du["edge_cost"], du["cloud_tokens"], du["cloud_cost"], preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": chunk_idx,
                        "status": "ok", "preview": preview, "error": None,
                        "wall_ms": m_wall_ms,
                        **du,
                    }
                except Exception as exc:
                    m_wall_ms = (time.monotonic() - m_t0) * 1000.0
                    u1 = usage_snapshot()
                    du = usage_diff(u0, u1)
                    logger.exception("Ingestion failed: conv %d %s chunk %d", conv_index, session_key, chunk_idx)
                    failed += 1
                    failures.append({
                        "session": session_key,
                        "chunk_index": chunk_idx,
                        "error": str(exc),
                    })
                    logger.info(
                        "[mem0 load] conv=%d %d/%d %s FAILED wall=%.2fs | %s",
                        conv_index, idx, total_chunks, session_key, m_wall_ms / 1000, preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": chunk_idx,
                        "status": "failed", "preview": preview, "error": str(exc)[:300],
                        "wall_ms": m_wall_ms,
                        **du,
                    }

        yield {
            "event": "done",
            "status": "success" if failed == 0 else "partial_failure",
            "conversation_id": conv_index,
            "user_id": user_id,
            "added": added,
            "failed": failed,
            "failures": failures if failures else None,
        }

    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compat wrapper: drains the iterator into a single dict."""
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
        """Trigger a no-op search so the Qdrant client establishes its
        connection / loads its collection metadata, paying that cost up
        front instead of folding it into row #1."""
        try:
            self.memory.search(query="warmup", user_id=f"warmup_{conv_index}", limit=1)
        except Exception:
            logger.exception("mem0 warmup search failed")
        return {"backend": "mem0", "warmed": True}

    def reset(self) -> Dict[str, Any]:
        """Drop the qdrant collection mem0 writes into so the next
        /memorize call starts on empty state. Mem0 hard-codes the
        collection name to `mem0` (configurable via
        MemoryConfig.collection_name, but we keep
        the default), so wiping it is the cleanest way to start fresh."""
        deleted = False
        try:
            from qdrant_client import QdrantClient
            qc = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
            for name in ("mem0", "mem0migrations"):
                if qc.collection_exists(name):
                    qc.delete_collection(name)
                    deleted = True
        except Exception:
            logger.exception("mem0 reset: qdrant collection drop failed")
        # Re-instantiate Memory so it lazily recreates collections on next add.
        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                },
            }
        )
        self.memory = Memory(config)
        self.run_id = uuid.uuid4().hex[:8]
        self.current_user_id = None
        return {"backend": "mem0", "deleted": deleted}

    def remember(self, question: str) -> List[str]:
        if not self.current_user_id:
            logger.warning("No active user_id — call memorize first.")
            return []

        logger.info("Mem0 search: query=%r user_id=%r limit=%d", question, self.current_user_id, self.TOP_K)
        try:
            search_results = self.memory.search(
                question,
                user_id=self.current_user_id,
                limit=self.TOP_K,
            )
        except Exception:
            logger.exception("Mem0 search failed")
            return []

        if isinstance(search_results, dict):
            memories = search_results.get("results", [])
        elif isinstance(search_results, list):
            memories = search_results
        else:
            memories = []

        results: List[str] = []
        for item in memories:
            if isinstance(item, dict):
                text = (item.get("memory") or "").strip()
                if text:
                    results.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    results.append(text)

        logger.info("Mem0 returned %d memories", len(results))
        return results
