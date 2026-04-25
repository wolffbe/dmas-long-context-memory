"""Memory-backend factory.

All four backends are built and started in parallel so each agent can answer
any incoming `/ask` regardless of which memory store the question targets.
A backend that fails to start is logged and replaced with the no-op `none`
backend for that slot — so a single broken store (e.g. unreachable qdrant)
doesn't take the agent down.
"""
import asyncio
import logging

from app.memory.base import MemoryBackend
from app.memory.none_backend import NoneBackend

logger = logging.getLogger("agent.memory")

BACKEND_NAMES = ("none", "mem0", "zep", "rag")


async def _build_one(name: str) -> MemoryBackend:
    if name == "mem0":
        from app.memory.mem0_backend import Mem0Backend
        b = Mem0Backend()
    elif name == "zep":
        from app.memory.zep_backend import ZepBackend
        b = ZepBackend()
    elif name == "rag":
        from app.memory.rag_backend import RagBackend
        b = RagBackend()
    elif name == "none":
        b = NoneBackend()
    else:
        raise ValueError(f"unknown backend: {name!r}")
    await b.start()
    return b


async def build_all_backends() -> dict[str, MemoryBackend]:
    """Start every backend in parallel. Tolerates per-backend failures."""
    async def _safe(name: str) -> tuple[str, MemoryBackend]:
        try:
            return name, await _build_one(name)
        except Exception as exc:
            logger.warning("backend %s failed to start: %s — falling back to none", name, exc)
            fallback = NoneBackend()
            await fallback.start()
            return name, fallback

    pairs = await asyncio.gather(*(_safe(n) for n in BACKEND_NAMES))
    return dict(pairs)
