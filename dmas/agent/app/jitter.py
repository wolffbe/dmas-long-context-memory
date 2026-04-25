"""Real round-trip latency probe to peer agents.

Pings each peer's `/health` (through that peer's toxiproxy) and returns the
mean wall-clock round-trip in milliseconds. The agent service uses this
BEFORE the LLM loop to decide whether to expose the `ask_peers` tool —
toxics applied per-/ask (latency, jitter, bandwidth) show up in this
measurement naturally.

Each probe is also recorded into the `peer_request_duration_seconds`
histogram (and failures into `peer_request_failures_total`) so Grafana sees
the per-/ask RTT alongside the heartbeat and fan-out samples.

Cached for 1 second so multi-tool-call paths don't hammer peers.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from app.config import CFG
from app.metrics import (
    peer_request_duration_seconds,
    peer_request_failures_total,
    src,
)


_CACHE: dict[str, tuple[float, float]] = {}
_TTL_S = 1.0
_TIMEOUT_S = 3.0


def _peer_id(peer_url: str) -> str:
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.isdigit() and len(token) >= 4 and token.startswith("180"):
            return token[-1]
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.startswith("agent-") and len(token) > 6:
            return token.split("-", 1)[1]
    return peer_url


async def _probe(client: httpx.AsyncClient, peer: str) -> float:
    url = f"{peer.rstrip('/')}/health"
    dst = _peer_id(peer)
    t0 = time.perf_counter()
    try:
        with peer_request_duration_seconds.labels(src=src(), dst=dst).time():
            await client.get(url, timeout=_TIMEOUT_S)
    except Exception:
        peer_request_failures_total.labels(src=src(), dst=dst).inc()
        # Failed probe: report the timeout as the observation rather than 0
        # so a downed peer doesn't look like "fast network".
        return _TIMEOUT_S * 1000.0
    return (time.perf_counter() - t0) * 1000.0


async def measured_peer_latency_ms() -> float:
    """Mean round-trip latency (ms) across all configured peers, cached 1s."""
    peers = list(CFG.peers)
    if not peers:
        return 0.0

    now = time.monotonic()
    cached = _CACHE.get("v")
    if cached and now - cached[0] < _TTL_S:
        return cached[1]

    async with httpx.AsyncClient() as client:
        samples = await asyncio.gather(*(_probe(client, p) for p in peers))
    mean = sum(samples) / len(samples) if samples else 0.0
    _CACHE["v"] = (now, mean)
    return mean


