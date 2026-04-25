from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class FullContextService:
    """Bypass backend: stores the raw conversation JSON and returns it
    verbatim. No LLM-driven extraction at memorize time, no retrieval at
    query time. The responder receives the full conversation JSON as its
    context and answers directly. Sets the lower bound on retention loss
    (none) and the upper bound on per-question context size."""

    def __init__(self):
        self._conversations: dict[int, Dict[str, Any]] = {}
        self.current_conv_index: int | None = None

    def memorize_iter(self, conv_index: int, data: Dict[str, Any]):
        sessions = data.get("sessions") or {}
        if not isinstance(sessions, dict):
            yield {"event": "done", "status": "error", "reason": "'sessions' must be a dict",
                   "added": 0, "failed": 0}
            return

        # The bench drives single-turn /memorize calls, so this accumulates
        # turns into the existing conversation rather than overwriting. The
        # full conversation only exists after the last call.
        existing = self._conversations.setdefault(conv_index, {})
        for k, v in data.items():
            if k == "sessions":
                continue
            existing.setdefault(k, v)
        merged_sessions = existing.setdefault("sessions", {})
        merged_dts = existing.setdefault("session_datetimes", {})
        for sk, turns_list in sessions.items():
            if not re.match(r"^session_\d+$", sk):
                continue
            if not isinstance(turns_list, list):
                continue
            merged_sessions.setdefault(sk, []).extend(turns_list)
        for k, v in (data.get("session_datetimes") or {}).items():
            merged_dts.setdefault(k, v)
        self.current_conv_index = conv_index

        session_keys = [k for k in sessions if re.match(r"^session_\d+$", k)]
        turns = sum(len(sessions[k]) for k in session_keys if isinstance(sessions[k], list))
        payload = json.dumps(existing, ensure_ascii=False)
        logger.info("FullContext append: conv=%d +sessions=%d +turns=%d total_chars=%d",
                    conv_index, len(session_keys), turns, len(payload))

        for sk in session_keys:
            yield {
                "event": "memory",
                "session": sk, "chunk_idx": 0, "status": "ok",
                "preview": json.dumps(sessions[sk], ensure_ascii=False)[:120].replace("\n", " "),
                "error": None,
                "wall_ms": 0.0,
                "edge_tokens": 0, "edge_cost": 0.0,
                "cloud_tokens": 0, "cloud_cost": 0.0,
            }

        yield {
            "event": "done",
            "status": "success",
            "conversation_id": conv_index,
            "added": 1,
            "failed": 0,
            "json_chars": len(payload),
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

    def reset(self) -> Dict[str, Any]:
        n = len(self._conversations)
        self._conversations.clear()
        self.current_conv_index = None
        return {"backend": "full_context", "deleted": n}

    def warmup(self, conv_index: int) -> Dict[str, Any]:
        # No persistent storage to bring up — included for schema parity.
        return {"backend": "full_context", "warmed": True}

    def remember(self, question: str) -> List[str]:
        if self.current_conv_index is None:
            logger.warning("No active conversation — call memorize first.")
            return []
        data = self._conversations.get(self.current_conv_index)
        if not data:
            return []
        return [json.dumps(data, ensure_ascii=False)]
