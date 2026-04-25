from typing import Any, Protocol


class MemoryBackend(Protocol):
    """All four backends conform to this minimal interface.

    `ingest` takes one LOCOMO-style message item carrying full metadata:
        {
          "conv_idx": int, "speaker_a": str, "speaker_b": str,
          "session_id": str, "session_datetime": str,    # raw LOCOMO date string
          "speaker": str, "text": str,
          "dia_id": str, "blip_caption": str | None, "query": str | None,
        }
    Each backend rebuilds the upstream-benchmark message shape internally.

    For LongMemEval the item carries:
        {
          "dataset": "longmemeval", "question_id": str,
          "session_id": str, "session_datetime": str,    # raw LongMemEval date string
          "role": "user"|"assistant", "content": str,
          "turn_index": int,
        }

    `recall(question, lookup_key)` returns a list of memory snippet strings.
    `lookup_key` is the LOCOMO `conv_idx` (int) or LongMemEval `question_id` (str).
    """

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def ingest(self, item: dict[str, Any]) -> dict[str, Any]: ...
    async def recall(self, question: str, lookup_key: str | int | None) -> list[str]: ...
