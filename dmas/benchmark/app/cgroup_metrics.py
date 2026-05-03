"""Direct cgroup-v2 read for zero-cache per-call metric attribution.

`/sys/fs/cgroup/system.slice/docker-<id>.scope/{cpu.stat, memory.current,
io.stat}` are kernel-maintained pseudo-files: every read reflects the
moment of access. No telegraf scrape interval, no docker daemon stats
cache. Two snapshots straddling a 200 ms `.add()` give a real 200 ms
delta.

cgroup v2 does not include network counters (those live per net-ns).
For network we read `/proc/<container_pid>/net/dev` from the host's
proc mount — also kernel-real-time.

Container labels (`group=edge|cloud`) and PIDs come from the docker
daemon via docker-proxy. Cached per process; the cache is rebuilt
whenever a labelled container's scope dir disappears (restart).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path(os.getenv("CGROUP_ROOT", "/host-cgroup"))
PROC_ROOT = Path(os.getenv("PROC_ROOT", "/host-proc"))
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")

# Pure pass-through proxies whose tx is a re-transmit of bytes the
# upstream container already counted. Including their tx in the group
# total would double-count every transfer that traverses them. Comma-
# separated container-name list; default skips toxiproxy.
NETWORK_TX_EXCLUDE = {
    n.strip() for n in os.getenv("NETWORK_TX_EXCLUDE", "toxiproxy").split(",") if n.strip()
}

# container_id -> {"group": "edge"|"cloud", "pid": str, "name": str}
_label_cache: dict[str, dict[str, str]] = {}
_docker_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    """Lazy unix-socket client to dockerd. No docker-proxy hop."""
    global _docker_client
    if _docker_client is None:
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
        _docker_client = httpx.AsyncClient(transport=transport, base_url="http://localhost")
    return _docker_client


async def _refresh_labels() -> None:
    c = _client()
    r = await c.get("/containers/json", timeout=3.0)
    r.raise_for_status()
    new: dict[str, dict[str, str]] = {}
    for entry in r.json():
        cid = entry.get("Id", "")
        if not cid:
            continue
        group = (entry.get("Labels") or {}).get("group")
        if group not in ("edge", "cloud"):
            continue
        # Names look like ["/toxiproxy"]; strip the leading slash.
        name = ""
        names = entry.get("Names") or []
        if names:
            name = names[0].lstrip("/")
        try:
            ri = await c.get(f"/containers/{cid}/json", timeout=3.0)
            pid = ri.json().get("State", {}).get("Pid", 0)
        except Exception as exc:
            logger.debug("inspect %s failed: %s", cid[:12], exc)
            pid = 0
        new[cid] = {"group": group, "pid": str(pid), "name": name}
    _label_cache.clear()
    _label_cache.update(new)
    logger.info("cgroup label cache: %d edge/cloud containers", len(new))


def _read_cpu_usec(scope: Path) -> int:
    try:
        for line in (scope / "cpu.stat").read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return 0


def _read_mem(scope: Path) -> int:
    try:
        return int((scope / "memory.current").read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return 0


def _read_mem_peak(scope: Path) -> int:
    """memory.peak is the high-water mark since the cgroup was created.
    Diffing two snapshots gives the *additional* peak attributable to the
    interval between them — i.e., the most RAM the cgroup needed to hold
    while the call was in flight, in excess of any prior peak."""
    try:
        return int((scope / "memory.peak").read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return 0


def _read_io(scope: Path) -> tuple[int, int]:
    rb = wb = 0
    try:
        for line in (scope / "io.stat").read_text().splitlines():
            for kv in line.split():
                if kv.startswith("rbytes="):
                    try: rb += int(kv.split("=", 1)[1])
                    except ValueError: pass
                elif kv.startswith("wbytes="):
                    try: wb += int(kv.split("=", 1)[1])
                    except ValueError: pass
    except (FileNotFoundError, PermissionError):
        pass
    return rb, wb


def _read_net_tx(pid: str) -> int:
    """Return tx_bytes for the container's network namespace.

    We only count tx (not rx) so a transfer A→B isn't counted twice
    (once as A.tx, once as B.rx). Each byte is attributed to its
    sender. Toxiproxy hops still count once per leg, which correctly
    reflects bandwidth actually used.
    """
    if not pid or pid == "0":
        return 0
    try:
        text = (PROC_ROOT / pid / "net" / "dev").read_text()
    except (FileNotFoundError, PermissionError):
        return 0
    tx = 0
    for line in text.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 17:
            continue
        iface = parts[0].rstrip(":")
        if iface == "lo":
            continue
        try:
            tx += int(parts[9])
        except ValueError:
            pass
    return tx


async def snapshot(_unused_client: httpx.AsyncClient | None = None) -> dict[str, float]:
    """Per-group sum of cgroup counters at the current instant.

    The httpx arg is unused (kept for API parity with the prior helper);
    docker queries go over the unix socket directly. Argument is renamed
    to avoid shadowing the module-level `_client()` factory.
    """
    if not _label_cache:
        try:
            await _refresh_labels()
        except Exception:
            logger.exception("docker label refresh failed")

    sums = {
        "cpu_edge_ns": 0, "cpu_cloud_ns": 0,
        "ram_edge_peak": 0, "ram_cloud_peak": 0,
        "disk_edge_read": 0, "disk_edge_write": 0,
        "disk_cloud_read": 0, "disk_cloud_write": 0,
        "net_edge_tx": 0, "net_cloud_tx": 0,
    }
    missing = False
    for cid, info in _label_cache.items():
        scope = CGROUP_ROOT / "system.slice" / f"docker-{cid}.scope"
        if not scope.exists():
            missing = True
            continue
        group = info["group"]
        cpu_us = _read_cpu_usec(scope)
        mem_peak = _read_mem_peak(scope)
        rb, wb = _read_io(scope)
        sums[f"cpu_{group}_ns"] += cpu_us * 1000  # usec -> ns
        sums[f"ram_{group}_peak"] += mem_peak
        sums[f"disk_{group}_read"] += rb
        sums[f"disk_{group}_write"] += wb
        # Network: tx-only across both groups so each byte is counted
        # once at its sender. Containers in NETWORK_TX_EXCLUDE
        # (toxiproxy) are skipped because they retransmit upstream
        # bytes verbatim, which would double-count them.
        if info.get("name", "") not in NETWORK_TX_EXCLUDE:
            sums[f"net_{group}_tx"] += _read_net_tx(info.get("pid", "0"))

    if missing:
        # A labelled container was restarted between snapshots — rebuild
        # for the next call so we don't keep skipping it.
        try:
            await _refresh_labels()
        except Exception:
            logger.exception("post-miss label refresh failed")

    return sums


async def wait_io_quiet(
    client: httpx.AsyncClient | None = None,
    *,
    poll_interval: float = 0.05,
    quiet_for: float = 0.2,
    max_wait: float = 5.0,
) -> dict[str, float]:
    """Poll cgroup snapshots until cloud disk I/O stops accumulating, then
    return the final snapshot. Lets per-row disk_cloud_bytes capture
    asynchronous DB flushes (Neo4j's checkpointer in particular) instead
    of attributing them to whichever row is unlucky enough to be in
    flight when the flush hits. Edge group is excluded from the quiescence
    check because Ollama doesn't flush asynchronously and we don't want
    to be held up by background activity from another container.

    Returns when delta(prev, curr) on disk_cloud_read+disk_cloud_write
    has been zero for `quiet_for` seconds, or when `max_wait` elapses."""
    s_prev = await snapshot(client)
    quiet_since = time.monotonic()
    deadline = quiet_since + max_wait
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        s_curr = await snapshot(client)
        d_disk = (
            (s_curr.get("disk_cloud_read", 0) - s_prev.get("disk_cloud_read", 0))
            + (s_curr.get("disk_cloud_write", 0) - s_prev.get("disk_cloud_write", 0))
        )
        if d_disk > 0:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= quiet_for:
            return s_curr
        s_prev = s_curr
    return s_prev


def delta(t0: dict[str, float], t1: dict[str, float]) -> dict[str, Any]:
    # disk_*_bytes is paired with `wait_io_quiet` on the call site so the
    # t1 snapshot is taken after async DB flushes (notably Neo4j's
    # checkpointer) settle. Without that wait, async writes spill onto
    # subsequent rows or fall outside the snapshot entirely; with it,
    # the per-row delta captures the call's true persistence cost.
    def d(k: str) -> float:
        return max(0.0, t1.get(k, 0.0) - t0.get(k, 0.0))
    # ram_*_peak_bytes is the *additional* high-water mark the cgroup hit
    # during this call — diff of memory.peak between two snapshots. Unlike
    # memory.current, this is monotonic, so the diff is non-negative and
    # tells you the worst-case working-set increase the call caused.
    return {
        "cpu_edge_ns":         d("cpu_edge_ns"),
        "cpu_cloud_ns":        d("cpu_cloud_ns"),
        "ram_edge_peak_bytes":  d("ram_edge_peak"),
        "ram_cloud_peak_bytes": d("ram_cloud_peak"),
        "disk_edge_bytes":     d("disk_edge_read") + d("disk_edge_write"),
        "disk_cloud_bytes":    d("disk_cloud_read") + d("disk_cloud_write"),
        "network_edge_bytes":  d("net_edge_tx"),
        "network_cloud_bytes": d("net_cloud_tx"),
    }
