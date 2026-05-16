from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Token budget for the JSON payload returned to the responder. The
# responder feeds this through gpt-4o-mini (128k context). Subtract
# responder overhead (system + date anchor + question + tool schema +
# message wrappers ≈ 260–460 tok) and reply headroom (~1000 tok); a
# 3000-token gap to the model boundary covers both with margin, so
# overflow becomes structurally impossible regardless of question
# length. Override via FULL_CONTEXT_MAX_TOKENS for a different model.
MAX_PAYLOAD_TOKENS = int(os.getenv("FULL_CONTEXT_MAX_TOKENS", "125000"))

# gpt-4o-mini uses o200k_base, NOT cl100k_base. Use the model-specific
# encoder so our count matches what OpenAI's tokenizer will report on
# the prompt — otherwise we'd over-count and truncate more than needed
# (safe direction, but wastes context).
try:
    import tiktoken
    _RESPONDER_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        _enc = tiktoken.encoding_for_model(_RESPONDER_MODEL)
    except KeyError:
        _enc = tiktoken.get_encoding("o200k_base")
    def _count_tokens(s: str) -> int:
        return len(_enc.encode(s, disallowed_special=()))
except Exception:
    # Fallback when tiktoken isn't installed: ~3 chars per token is
    # conservative for JSON. Better to truncate slightly more than to
    # overflow the model.
    def _count_tokens(s: str) -> int:
        return len(s) // 3


class FullContextService:
    """Bypass backend: stores the raw conversation JSON and returns it
    verbatim. No LLM-driven extraction at memorize time, no retrieval at
    query time. The responder receives the full conversation JSON as its
    context and answers directly. Sets the lower bound on retention loss
    (none) and the upper bound on per-question context size.

    On `remember`, if the serialised conversation exceeds the responder
    model's context window, the oldest turns are dropped until the
    payload fits. The truncation marker is included in the returned
    JSON so traces show exactly how many turns were sacrificed."""

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

        serialized = json.dumps(data, ensure_ascii=False)
        n_tokens = _count_tokens(serialized)
        if n_tokens <= MAX_PAYLOAD_TOKENS:
            return [serialized]

        # Overflow path: compute per-turn token cost once, then drop
        # exactly enough oldest turns to fit. Single pass — no iterative
        # re-tokenization of the full payload.
        work: Dict[str, Any] = {k: v for k, v in data.items()
                                if k not in ("sessions", "session_datetimes")}
        work["sessions"] = {sk: list(v) for sk, v in (data.get("sessions") or {}).items()}
        work["session_datetimes"] = dict(data.get("session_datetimes") or {})

        def _snum(k: str) -> int:
            try:
                return int(k.split("_", 1)[1])
            except Exception:
                return 0
        keys = sorted(work["sessions"].keys(), key=_snum)

        # Tokens to shed. Add a small safety margin (1%) — per-turn cost
        # measured in isolation slightly undercounts the cost in-context
        # because JSON separators between turns are not per-turn.
        excess = n_tokens - MAX_PAYLOAD_TOKENS
        target_drop = int(excess * 1.01) + 1

        dropped_turns = 0
        dropped_tokens = 0
        while keys and dropped_tokens < target_drop:
            sess = keys[0]
            if work["sessions"][sess]:
                turn = work["sessions"][sess].pop(0)
                dropped_tokens += _count_tokens(json.dumps(turn, ensure_ascii=False))
                dropped_turns += 1
            else:
                del work["sessions"][sess]
                work["session_datetimes"].pop(sess, None)
                keys.pop(0)

        # One final verification. The per-turn-in-isolation estimate is
        # tight in practice; this loop typically runs zero or one extra
        # drop. Belt-and-suspenders so we never return an overflowing
        # payload even if the estimate happened to undershoot.
        while keys and _count_tokens(json.dumps(work, ensure_ascii=False)) > MAX_PAYLOAD_TOKENS:
            sess = keys[0]
            if work["sessions"][sess]:
                work["sessions"][sess].pop(0)
                dropped_turns += 1
            else:
                del work["sessions"][sess]
                work["session_datetimes"].pop(sess, None)
                keys.pop(0)

        work["_truncated"] = {
            "dropped_turns_from_oldest": dropped_turns,
            "max_tokens": MAX_PAYLOAD_TOKENS,
            "original_tokens": n_tokens,
        }
        final = json.dumps(work, ensure_ascii=False)
        logger.info("FullContext truncate: conv=%d %d->%d tokens (dropped %d oldest turns)",
                    self.current_conv_index, n_tokens, _count_tokens(final), dropped_turns)
        return [final]
