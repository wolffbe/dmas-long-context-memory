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
    trace_id: str | None = None,
    session_id: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"backend": backend, "conv_index": conv_index, "data": data}
    if trace_id:
        payload["trace_id"] = trace_id
    if session_id is not None:
        payload["session_id"] = session_id
    if mode is not None:
        payload["mode"] = mode
    r = await client.post(
        f"{COORDINATOR_URL}/memorize",
        json=payload,
        timeout=httpx.Timeout(3600.0),
    )
    r.raise_for_status()
    return r.json()


async def post_ask(client: httpx.AsyncClient, question: str, backend: str,
                   session_date: str = "",
                   session_id: str | None = None,
                   conv_index: int | None = None,
                   mode: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question, "backend": backend,
                               "session_date": session_date}
    if session_id is not None:
        payload["session_id"] = session_id
    if conv_index is not None:
        payload["conv_index"] = conv_index
    if mode is not None:
        payload["mode"] = mode
    r = await client.post(
        f"{COORDINATOR_URL}/ask",
        json=payload,
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
