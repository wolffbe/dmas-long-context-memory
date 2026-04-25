"""Async clients for talking to the per-agent services from the coordinator."""
from __future__ import annotations

from typing import Any

import httpx


async def post_load(client: httpx.AsyncClient, agent_url: str,
                    items: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    r = await client.post(
        f"{agent_url}/admin/load",
        json={"items": items, "backend": backend},
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


async def post_ask(
    client: httpx.AsyncClient, agent_url: str,
    *,
    question: str, backend: str, llm_model: str,
    models: dict[str, str],
    conv_idx: int | None = None,
    question_id: str | None = None,
    peer_latency_threshold_ms: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "backend": backend,
        "llm_model": llm_model,
        "models": models,
    }
    if conv_idx is not None:
        payload["conv_idx"] = conv_idx
    if question_id is not None:
        payload["question_id"] = question_id
    if peer_latency_threshold_ms is not None:
        payload["peer_latency_threshold_ms"] = peer_latency_threshold_ms
    r = await client.post(f"{agent_url}/ask", json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()
