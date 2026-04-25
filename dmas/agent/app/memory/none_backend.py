from typing import Any


class NoneBackend:
    async def start(self) -> None: ...
    async def close(self) -> None: ...

    async def ingest(self, item: dict[str, Any]) -> dict[str, Any]:
        return {"stored": False, "reason": "no-memory backend"}

    async def recall(self, question: str, lookup_key: str | int | None) -> list[str]:
        return []
