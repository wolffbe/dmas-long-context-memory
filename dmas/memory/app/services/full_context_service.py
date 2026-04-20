import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

MAX_CHARS = int(os.getenv("FULL_CONTEXT_MAX_CHARS", "400000"))


class FullContextService:
    """No-memory baseline: stores entire conversation in RAM and returns it verbatim.

    Mirrors the Full-Context approach from the Mem0 paper (~73% accuracy, high cost).
    No external database is used; state lives in-process per conv_index.
    """

    def __init__(self):
        self._store: Dict[int, str] = {}

    def memorize_conversation(self, conv_index: int, data: Dict[str, Any]) -> Dict[str, Any]:
        sessions = data.get("sessions") or {}
        session_datetimes = data.get("session_datetimes") or {}

        if not isinstance(sessions, dict):
            return {"status": "error", "reason": "'sessions' must be a dict"}

        lines: List[str] = []
        for session_key in sorted(sessions.keys()):
            turns = sessions[session_key]
            if not isinstance(turns, list):
                continue
            ts = session_datetimes.get(f"{session_key}_date_time", "")
            ts_str = f" [{ts}]" if ts else ""
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text", "")).strip()
                speaker = str(turn.get("speaker", "")).strip()
                if not text:
                    continue
                prefix = f"{speaker}{ts_str}" if speaker else f"Unknown{ts_str}"
                lines.append(f"[{prefix}]: {text}")

        new_text = "\n".join(lines)
        existing = self._store.get(conv_index, "")
        combined = (existing + "\n" + new_text).strip() if existing else new_text
        self._store[conv_index] = combined

        logger.info(
            "FullContext: conv %d stored %d chars total (%d turns added)",
            conv_index, len(combined), len(lines),
        )
        return {
            "status": "success",
            "conversation_id": conv_index,
            "turns_added": len(lines),
            "total_chars": len(combined),
        }

    def remember(self, question: str) -> List[str]:
        if not self._store:
            logger.warning("FullContext: no conversations loaded")
            return []
        full_text = "\n\n".join(self._store.values())
        if len(full_text) > MAX_CHARS:
            full_text = full_text[:MAX_CHARS]
            logger.warning("FullContext: context truncated to %d chars", MAX_CHARS)
        return [full_text] if full_text else []
