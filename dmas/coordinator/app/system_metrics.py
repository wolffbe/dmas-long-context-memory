"""Per-question Prometheus snapshots — async, lives in the coordinator.

Edge   = agent-1 + litellm-1 + toxiproxy-1 + qdrant-1 + neo4j-1 + ollama + coordinator
Cloud  = agent-2/3 + litellm-2/3 + toxiproxy-2/3 + qdrant-2/3 + neo4j-2/3
Observability (langfuse-*) and the monitoring stack are excluded — they
have no `group` label.

Resolution floor: telegraf gathers docker stats every 5s, Prometheus scrapes
telegraf every 5s. Sub-5s per-question deltas will be noisy or zero.

Methodology
-----------
- **CPU** (`cpu_*_ns`): sum of cgroup `docker_container_cpu_usage_total`
  nanoseconds across the group, t1 − t0. Each container's counter is
  independent, so summing does not double-count.

- **RAM** (`ram_*_bytes`): peak resident set size during the question,
  computed at t1 as `max_over_time(sum(docker_container_mem_usage[<wall+10>s:5s]))`.
  Resolves the previous "single t1 gauge sample" miss for sub-5s questions.

- **Disk** (`disk_*_bytes`): sum of cgroup blkio
  `io_service_bytes_recursive_read + write`, t1 − t0. blkio is per-cgroup
  kernel I/O, so summing across containers does not double-count.

- **Network** (`network_*_bytes`): sum of `docker_container_net_tx_bytes`
  across the group — TX ONLY, no exclusions. Counting tx-only counts each
  wire transmission exactly once (each veth pair's tx counter is paired
  with the other end's rx, so adding rx would double the same hop).
  Multi-hop chains (e.g. agent → toxiproxy → peer, or agent → litellm →
  external) get one count per hop, which is what we want — each hop is
  a distinct wire transmission consuming real network resources at that
  layer. Ollama is included because it's part of the edge subnet and
  originates response bytes back to agent-1; toxiproxy and litellm are
  included because their forwarding work is real per-hop network use.
  All traffic is captured, no wire-byte is counted twice.
"""
from __future__ import annotations

import asyncio

import httpx

from app.config import CFG


_COUNTER_QUERIES: dict[str, str] = {
    "cpu_edge_ns":   'sum(docker_container_cpu_usage_total{group="edge"})',
    "cpu_cloud_ns":  'sum(docker_container_cpu_usage_total{group="cloud"})',
    "disk_edge_read_bytes":   'sum(docker_container_blkio_io_service_bytes_recursive_read{group="edge"})',
    "disk_edge_write_bytes":  'sum(docker_container_blkio_io_service_bytes_recursive_write{group="edge"})',
    "disk_cloud_read_bytes":  'sum(docker_container_blkio_io_service_bytes_recursive_read{group="cloud"})',
    "disk_cloud_write_bytes": 'sum(docker_container_blkio_io_service_bytes_recursive_write{group="cloud"})',
    "net_edge_tx":   'sum(docker_container_net_tx_bytes{group="edge"})',
    "net_cloud_tx":  'sum(docker_container_net_tx_bytes{group="cloud"})',
}


async def _instant(client: httpx.AsyncClient, query: str) -> float:
    try:
        r = await client.get(
            f"{CFG.prometheus_url}/api/v1/query",
            params={"query": query}, timeout=5.0,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("status") != "success":
            return 0.0
        result = body["data"]["result"]
        if not result:
            return 0.0
        return float(result[0]["value"][1])
    except Exception:
        return 0.0


async def snapshot(client: httpx.AsyncClient) -> dict[str, float]:
    """Instant query of all monotonic counters. RAM is handled separately
    by `peak_ram` after the question completes."""
    keys, queries = zip(*_COUNTER_QUERIES.items())
    vals = await asyncio.gather(*(_instant(client, q) for q in queries))
    return dict(zip(keys, vals))


async def peak_ram(client: httpx.AsyncClient, window_s: float) -> dict[str, float]:
    """Max RSS each group reached during the question.

    Evaluated at "now" (= t1), looking back over the question's wall time
    plus a 10s buffer (telegraf scrapes every 5s — without slack a 1-2s
    question can fall between two scrapes and miss the peak entirely).
    """
    span = max(int(window_s) + 10, 30)
    queries = {
        "ram_edge_bytes":  f'max_over_time(sum(docker_container_mem_usage{{group="edge"}})[{span}s:5s])',
        "ram_cloud_bytes": f'max_over_time(sum(docker_container_mem_usage{{group="cloud"}})[{span}s:5s])',
    }
    keys, qs = zip(*queries.items())
    vals = await asyncio.gather(*(_instant(client, q) for q in qs))
    return dict(zip(keys, vals))


def delta(
    t0: dict[str, float], t1: dict[str, float],
    ram_peak: dict[str, float] | None = None,
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for k in _COUNTER_QUERIES:
        raw[k] = max(0.0, t1.get(k, 0.0) - t0.get(k, 0.0))
    ram_peak = ram_peak or {}
    return {
        "cpu_edge_ns":         raw["cpu_edge_ns"],
        "cpu_cloud_ns":        raw["cpu_cloud_ns"],
        "ram_edge_bytes":      ram_peak.get("ram_edge_bytes", 0.0),
        "ram_cloud_bytes":     ram_peak.get("ram_cloud_bytes", 0.0),
        "disk_edge_bytes":     raw["disk_edge_read_bytes"] + raw["disk_edge_write_bytes"],
        "disk_cloud_bytes":    raw["disk_cloud_read_bytes"] + raw["disk_cloud_write_bytes"],
        "network_edge_bytes":  raw["net_edge_tx"],
        "network_cloud_bytes": raw["net_cloud_tx"],
    }
