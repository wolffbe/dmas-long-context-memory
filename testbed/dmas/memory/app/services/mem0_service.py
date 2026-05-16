from __future__ import annotations

import contextvars
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from mem0 import Memory
from mem0.configs import prompts as _mem0_prompts
from mem0.configs.base import MemoryConfig

from shared.errors import exc_trace
from shared.litellm_usage import usage_snapshot_sync as usage_snapshot, diff as usage_diff
from shared.openai_warmup import warmup_chat_sync

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1

# Per-call session-date anchor for mem0's fact extractor.
#
# Why a ContextVar + monkey-patch: mem0's additive extraction prompt
# stamps `## Current Date` via `_resolve_dates()`, which defaults to
# `datetime.now()` because `Memory.add()` does NOT forward our
# `metadata={"timestamp": ...}` into `generate_additive_extraction_prompt`
# (mem0 v1.x). With a 2026-running container ingesting a 2023 LoCoMo
# conversation, the LLM's anchor is "today" and it emits facts like
# `"On 2026-05-02, User Caroline ..."` — wrong by years.
#
# We bind the active session's ISO date into this var before every
# `memory.add()` and restore the original `_resolve_dates` afterwards;
# the patched version reads the var and pins both the current and
# observation dates to it, so the extractor's anchor matches the
# conversation period.
_active_session_date: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_session_date", default=None,
)


# Idempotent: re-importing this module (e.g. in test harnesses) must not
# stack patches and must not lose the original function reference.
if not getattr(_mem0_prompts, "_dmas_patched", False):
    _orig_resolve_dates = _mem0_prompts._resolve_dates

    def _patched_resolve_dates(current_date=None, observation_date=None):
        pinned = _active_session_date.get()
        if pinned and current_date is None:
            current_date = pinned
        return _orig_resolve_dates(current_date, observation_date)

    _mem0_prompts._resolve_dates = _patched_resolve_dates
    _mem0_prompts._dmas_patched = True


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


def session_to_chunks(turns: list[dict], speaker_a: str, speaker_b: str,
                      session_date: str = "") -> list[list[dict]]:
    """Convert turns to message chunks for ingestion.

    `session_date` is the LOCOMO `session_*_date_time` string. When
    parseable we prepend the ISO 8601 UTC timestamp in brackets to every
    message's `content`, e.g. `"[2023-05-08T13:56:00+00:00] Caroline: ..."`.
    Mem0 OSS `Memory.add()` has no dedicated `created_at` / `reference_time`
    kwarg (only the cloud V3 API does), so the message body is the only
    channel that actually reaches the extractor — anchoring it explicitly
    here matches what cognee does and gives the LLM a concrete reference
    time for relative phrases like "yesterday".
    """
    parsed = parse_locomo_date(session_date)
    iso_prefix = parsed.replace(tzinfo=timezone.utc).isoformat() if parsed else None

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
        body = f"{speaker}: {text}"
        if iso_prefix:
            body = f"[{iso_prefix}] {body}"
        messages.append({"role": role, "content": body})

    chunks = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i : i + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
    return chunks


class Mem0Service:

    def __init__(self):
        self.TOP_K = int(os.getenv("MEMORIES_SEARCH_LIMIT", "20"))
        self.memory = Memory(self._build_config())
        self.run_id = uuid.uuid4().hex[:8]
        self.current_user_id: str | None = None

    @staticmethod
    def _build_config() -> MemoryConfig:
        return MemoryConfig(
            llm={
                "provider": "openai",
                "config": {
                    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    "temperature": 0,
                },
            },
            vector_store={
                "provider": "qdrant",
                "config": {
                    "host": os.getenv("QDRANT_HOST", "localhost"),
                    "port": int(os.getenv("QDRANT_PORT", "6333")),
                },
            },
        )

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
        total_chunks = sum(len(session_to_chunks(s, speaker_a, speaker_b, d)) for _, d, s in sorted_sessions)

        logger.info(
            "[mem0 load] conv=%d %s & %s sessions=%d chunks=%d",
            conv_index, speaker_a, speaker_b, len(sorted_sessions), total_chunks,
        )

        added = 0
        failed = 0
        failures: List[Dict[str, Any]] = []

        for session_key, date_str, turns in sorted_sessions:
            chunks = session_to_chunks(turns, speaker_a, speaker_b, date_str)
            if not chunks:
                continue

            session_epoch = locomo_date_to_epoch(date_str)
            parsed_session_date = parse_locomo_date(date_str)
            session_iso_date = (
                parsed_session_date.replace(tzinfo=timezone.utc).date().isoformat()
                if parsed_session_date else None
            )

            for chunk_idx, messages in enumerate(chunks):
                if any(not msg.get("content", "").strip() for msg in messages):
                    continue

                idx = added + failed + 1
                preview = (messages[0].get("content", "") if messages else "")[:120].replace("\n", " ")

                m_t0 = time.monotonic()
                u0 = usage_snapshot()
                token = _active_session_date.set(session_iso_date)
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
                    err_compact = exc_trace(exc)
                    failures.append({
                        "session": session_key,
                        "chunk_index": chunk_idx,
                        "error": err_compact,
                    })
                    logger.info(
                        "[mem0 load] conv=%d %d/%d %s FAILED wall=%.2fs | %s",
                        conv_index, idx, total_chunks, session_key, m_wall_ms / 1000, preview,
                    )
                    yield {
                        "event": "memory",
                        "session": session_key, "chunk_idx": chunk_idx,
                        "status": "failed", "preview": preview,
                        "error": err_compact[:300],
                        "wall_ms": m_wall_ms,
                        **du,
                    }
                finally:
                    _active_session_date.reset(token)

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
        """Pay mem0's one-time setup costs up front: Qdrant client
        connect + collection-metadata load, then a litellm chat ping so
        row #1 of the load loop doesn't pay the cold OpenAI handshake."""
        try:
            # mem0 v2 moved entity scoping into `filters=`; passing
            # user_id at the top level now raises ValueError.
            self.memory.search(
                query="warmup",
                filters={"user_id": f"warmup_{conv_index}"},
                limit=1,
            )
        except Exception:
            logger.exception("mem0 warmup search failed")
        try:
            warmup_chat_sync()
        except Exception:
            logger.exception("mem0 warmup chat.completions failed")
        return {"backend": "mem0", "warmed": True}

    def reset(self) -> Dict[str, Any]:
        """Drop the qdrant collection mem0 writes into so the next
        /memorize call starts on empty state. Mem0 hard-codes the
        collection name to `mem0` (configurable via
        MemoryConfig.collection_name, but we keep
        the default), so wiping it is the cleanest way to start fresh.
        Also drop any aliases pointing at mem0 collections — aliases
        survive collection deletion otherwise."""
        deleted = False
        dropped_aliases = 0
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
            try:
                for a in qc.get_aliases().aliases:
                    if a.collection_name in ("mem0", "mem0migrations"):
                        try:
                            qc.delete_collection_alias(a.alias_name)
                            dropped_aliases += 1
                        except Exception:
                            logger.exception("mem0 reset: drop alias %s failed", a.alias_name)
            except Exception:
                logger.exception("mem0 reset: list aliases failed")
        except Exception:
            logger.exception("mem0 reset: qdrant collection drop failed")
        # Re-instantiate Memory so it lazily recreates collections on next add.
        self.memory = Memory(self._build_config())
        self.run_id = uuid.uuid4().hex[:8]
        self.current_user_id = None
        return {"backend": "mem0", "deleted": deleted,
                "qdrant_aliases_dropped": dropped_aliases}

    def remember(self, question: str) -> List[str]:
        if not self.current_user_id:
            logger.warning("No active user_id — call memorize first.")
            return []

        logger.info("Mem0 search: query=%r user_id=%r limit=%d", question, self.current_user_id, self.TOP_K)
        try:
            # mem0 v2 moved entity scoping into `filters=`; passing
            # user_id at the top level now raises ValueError.
            search_results = self.memory.search(
                question,
                filters={"user_id": self.current_user_id},
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
