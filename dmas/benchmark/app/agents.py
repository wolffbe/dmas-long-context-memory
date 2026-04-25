"""HTTP wrappers for the benchmark's downstream coordinator.

Both memorize and ask go through the slim coordinator, so load-phase and
ask-phase traffic experience the same network conditions (toxiproxy is
between coordinator and memory/responder). The request body decides which
backend (mem0|graphiti|rag|full_context) to use.
"""
from __future__ import annotations

from typing import Any

import httpx

COORDINATOR_URL = "http://coordinator:8001"


async def post_memorize(
    client: httpx.AsyncClient,
    backend: str,
    conv_index: int,
    data: dict[str, Any],
) -> dict[str, Any]:
    r = await client.post(
        f"{COORDINATOR_URL}/memorize",
        json={"backend": backend, "conv_index": conv_index, "data": data},
        timeout=httpx.Timeout(3600.0),
    )
    r.raise_for_status()
    return r.json()


async def post_ask(client: httpx.AsyncClient, question: str, backend: str) -> dict[str, Any]:
    r = await client.post(
        f"{COORDINATOR_URL}/ask",
        json={"question": question, "backend": backend},
        timeout=httpx.Timeout(600.0),
    )
    r.raise_for_status()
    return r.json()


async def post_reset(client: httpx.AsyncClient, backend: str) -> dict[str, Any]:
    r = await client.post(
        f"{COORDINATOR_URL}/reset",
        json={"backend": backend},
        timeout=httpx.Timeout(600.0),
    )
    r.raise_for_status()
    return r.json()


async def post_warmup(client: httpx.AsyncClient, backend: str, conv_index: int) -> dict[str, Any]:
    r = await client.post(
        f"{COORDINATOR_URL}/warmup",
        json={"backend": backend, "conv_index": conv_index},
        timeout=httpx.Timeout(600.0),
    )
    r.raise_for_status()
    return r.json()
