"""Background peer-heartbeat task.

Each agent periodically calls every configured peer's `/health` endpoint via
its toxiproxy peer URL. This:
  * populates `peer_request_duration_seconds` (drives the Grafana inter-agent
    latency dashboard) even with no /ask traffic
  * generates `toxiproxy_proxy_*_bytes_total` traffic on the peer paths

Started in main.py during FastAPI lifespan startup.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from app.config import CFG
from app.metrics import (
    peer_request_duration_seconds,
    peer_request_failures_total,
    peer_toxic_jitter_ms,
    peer_toxic_latency_ms,
)

log = logging.getLogger(__name__)

INTERVAL_S = float(os.getenv("HEARTBEAT_INTERVAL_S", "10"))
TIMEOUT_S = 3.0


def _src() -> str:
    return CFG.agent_id


def _peer_id(peer_url: str) -> str:
    """Extract destination agent id from a peer URL.

    Peer URLs go through the agent's own toxiproxy on a per-destination port:
    18001 → agent-1, 18002 → agent-2, 18003 → agent-3. So the destination is
    encoded in the LAST digit of the proxy port, not the toxiproxy host.
    Falls back to host-based parsing if it's a direct agent-N URL.
    """
    # Try port → agent-id (the convention used by toxiproxy peer proxies)
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.isdigit() and len(token) >= 4 and token.startswith("180"):
            return token[-1]
    # Fall back to direct host parse (agent-N or toxiproxy-N)
    for token in peer_url.replace("/", " ").replace(":", " ").split():
        if token.startswith("agent-") and len(token) > 6:
            return token.split("-", 1)[1]
    return peer_url


async def _heartbeat_loop() -> None:
    if not CFG.peers:
        log.info("heartbeat disabled: no peers configured")
        return
    log.info(f"heartbeat: pinging {len(CFG.peers)} peers every {INTERVAL_S:.1f}s")
    src = _src()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        while True:
            await asyncio.gather(
                _refresh_toxic_gauges(client, src),
                *(_ping(client, src, p) for p in CFG.peers),
                return_exceptions=True,
            )
            await asyncio.sleep(INTERVAL_S)


async def _refresh_toxic_gauges(client: httpx.AsyncClient, src: str) -> None:
    """Read this agent's toxiproxy admin and publish current latency/jitter
    as Prometheus gauges. 0/0 when no toxic is set."""
    if not CFG.toxiproxy_admin:
        return
    url = f"{CFG.toxiproxy_admin.rstrip('/')}/proxies/{CFG.jitter_proxy_name}/toxics"
    latency = 0.0
    jitter = 0.0
    try:
        r = await client.get(url)
        r.raise_for_status()
        for t in r.json():
            if t.get("type") == "latency":
                attrs = t.get("attributes") or {}
                latency = float(attrs.get("latency", 0))
                jitter = float(attrs.get("jitter", 0))
                break
    except Exception:
        pass
    peer_toxic_latency_ms.labels(src=src).set(latency)
    peer_toxic_jitter_ms.labels(src=src).set(jitter)


async def _ping(client: httpx.AsyncClient, src: str, peer_url: str) -> None:
    dst = _peer_id(peer_url)
    url = f"{peer_url.rstrip('/')}/health"
    with peer_request_duration_seconds.labels(src=src, dst=dst).time():
        try:
            r = await client.get(url)
            r.raise_for_status()
        except Exception:
            peer_request_failures_total.labels(src=src, dst=dst).inc()


def start() -> asyncio.Task:
    return asyncio.create_task(_heartbeat_loop(), name="peer-heartbeat")
