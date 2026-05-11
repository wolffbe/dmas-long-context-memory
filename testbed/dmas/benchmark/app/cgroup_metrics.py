"""Direct cgroup-v2 read for zero-cache per-call metric attribution.

`/sys/fs/cgroup/system.slice/docker-<id>.scope/{cpu.stat, memory.current,
io.stat}` are kernel-maintained pseudo-files: every read reflects the
moment of access. No telegraf scrape interval, no docker daemon stats
cache. Two snapshots straddling a 200 ms `.add()` give a real 200 ms
delta.

CPU/RAM/disk are per-container, summed by `group=edge|cloud` label.

Network is measured at the single edge↔cloud gateway (`toxiproxy`): its
edge-net interface is the only path between the two data-plane subnets
by construction (responder↔memory go direct on cloud-net, OTel/admin
go over mgmt-net). One pair of counters — rx_bytes (edge→cloud) and
tx_bytes (cloud→edge) — replaces the prior per-container tx sum, which
double-counted intra-cloud chatter and was inflated by storage traffic.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CGROUP_ROOT = Path(os.getenv("CGROUP_ROOT", "/host-cgroup"))
PROC_ROOT = Path(os.getenv("PROC_ROOT", "/host-proc"))
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")

# Gateway-veth metering: the bench finds this container, reads its
# /proc/<pid>/net/route to identify the interface routing to
# EDGE_NET_CIDR, then reads rx/tx on that interface for the edge↔cloud
# counter. Defaults match docker-compose.yml.
GATEWAY_CONTAINER = os.getenv("GATEWAY_CONTAINER", "toxiproxy")
EDGE_NET_CIDR = os.getenv("EDGE_NET_CIDR", "172.30.1.0/24")

# container_id -> {"group": "edge"|"cloud", "pid": str, "name": str}
_label_cache: dict[str, dict[str, str]] = {}
# Cached (pid, iface_name) for the gateway. Invalidated when the gateway
# container is restarted (scope dir disappears) — same trigger as the
# label cache rebuild.
_gateway_cache: dict[str, str] | None = None
_docker_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    """Lazy unix-socket client to dockerd. No docker-proxy hop."""
    global _docker_client
    if _docker_client is None:
        transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
        _docker_client = httpx.AsyncClient(transport=transport, base_url="http://localhost")
    return _docker_client


async def _refresh_labels() -> None:
    global _gateway_cache
    c = _client()
    r = await c.get("/containers/json", timeout=3.0)
    r.raise_for_status()
    new: dict[str, dict[str, str]] = {}
    gateway_cid: str | None = None
    gateway_pid: str | None = None
    for entry in r.json():
        cid = entry.get("Id", "")
        if not cid:
            continue
        # Names look like ["/toxiproxy"]; strip the leading slash.
        name = ""
        names = entry.get("Names") or []
        if names:
            name = names[0].lstrip("/")
        group = (entry.get("Labels") or {}).get("group")
        is_gateway = name == GATEWAY_CONTAINER
        if group not in ("edge", "cloud") and not is_gateway:
            continue
        try:
            ri = await c.get(f"/containers/{cid}/json", timeout=3.0)
            pid = ri.json().get("State", {}).get("Pid", 0)
        except Exception as exc:
            logger.debug("inspect %s failed: %s", cid[:12], exc)
            pid = 0
        if is_gateway:
            gateway_cid = cid
            gateway_pid = str(pid)
        if group in ("edge", "cloud"):
            new[cid] = {"group": group, "pid": str(pid), "name": name}
    _label_cache.clear()
    _label_cache.update(new)
    # Resolve the gateway's edge-net interface by route inspection. The
    # iface name + container id are cached so steady-state snapshots only
    # touch /proc/<pid>/net/dev for the one interface we care about, and
    # the cid lets us detect gateway restarts via the scope-dir check.
    _gateway_cache = None
    if gateway_pid and gateway_pid != "0":
        iface = _resolve_gateway_iface(gateway_pid, EDGE_NET_CIDR)
        if iface:
            _gateway_cache = {"cid": gateway_cid, "pid": gateway_pid, "iface": iface}
            logger.info("gateway %s pid=%s edge-iface=%s", GATEWAY_CONTAINER, gateway_pid, iface)
        else:
            logger.warning("gateway %s pid=%s: no interface routes to %s — boundary counter will be 0",
                           GATEWAY_CONTAINER, gateway_pid, EDGE_NET_CIDR)
    logger.info("cgroup label cache: %d edge/cloud containers (gateway %s)",
                len(new), "found" if _gateway_cache else "missing")


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


def _resolve_gateway_iface(pid: str, cidr: str) -> str | None:
    """Find the interface inside `pid`'s netns that routes to `cidr`.

    `/proc/<pid>/net/route` columns: `Iface Destination Gateway Flags
    RefCnt Use Metric Mask MTU Window IRTT`. Destination and Mask are
    32-bit big-endian numbers serialized as little-endian hex (the
    kernel writes them via `seq_printf("%08X", ...)` over the native-
    order integer, which on x86_64 is little-endian). Decode each row,
    OR dest with mask, and match against the target subnet.
    """
    try:
        text = (PROC_ROOT / pid / "net" / "route").read_text()
    except (FileNotFoundError, PermissionError):
        return None
    try:
        target = ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        logger.warning("EDGE_NET_CIDR %r is not a valid CIDR", cidr)
        return None
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        iface, dest_hex, _gw, _flags, _r, _u, _m, mask_hex = parts[:8]
        try:
            dest = socket.inet_ntoa(struct.pack("<I", int(dest_hex, 16)))
            mask = socket.inet_ntoa(struct.pack("<I", int(mask_hex, 16)))
        except (ValueError, struct.error):
            continue
        try:
            row_net = ipaddress.ip_network(f"{dest}/{mask}", strict=False)
        except ValueError:
            continue
        if row_net == target:
            return iface
    return None


def _read_iface_bytes(pid: str, iface: str) -> tuple[int, int]:
    """Return (rx_bytes, tx_bytes) for `iface` in `pid`'s netns.

    Format of /proc/<pid>/net/dev:
        Inter-|   Receive                                                |  Transmit
         face |bytes    packets errs drop fifo frame compressed multicast|bytes ...
        eth0:   123456    789 ...                                          1000 ...

    Columns: 0=iface (with trailing colon), 1=rx_bytes, 9=tx_bytes.
    """
    if not pid or pid == "0":
        return 0, 0
    try:
        text = (PROC_ROOT / pid / "net" / "dev").read_text()
    except (FileNotFoundError, PermissionError):
        return 0, 0
    for line in text.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 17:
            continue
        if parts[0].rstrip(":") != iface:
            continue
        try:
            return int(parts[1]), int(parts[9])
        except ValueError:
            return 0, 0
    return 0, 0


async def snapshot(_unused_client: httpx.AsyncClient | None = None) -> dict[str, float]:
    """Per-group sum of cgroup counters at the current instant.

    The httpx arg is unused (kept for API parity with the prior helper);
    docker queries go over the unix socket directly. Argument is renamed
    to avoid shadowing the module-level `_client()` factory.
    """
    if not _label_cache or _gateway_cache is None:
        # Refresh whenever the gateway hasn't been resolved yet, even if
        # we already have labeled containers cached. Otherwise a bench
        # start before toxiproxy is ready leaves the boundary counter
        # stuck at 0 for the rest of the run.
        try:
            await _refresh_labels()
        except Exception:
            logger.exception("docker label refresh failed")

    sums = {
        "cpu_edge_ns": 0, "cpu_cloud_ns": 0,
        "ram_edge_peak": 0, "ram_cloud_peak": 0,
        "disk_edge_read": 0, "disk_edge_write": 0,
        "disk_cloud_read": 0, "disk_cloud_write": 0,
        # Gateway-veth rx/tx on toxiproxy's edge-net interface. rx = bytes
        # arriving from edge (edge→cloud), tx = bytes leaving toward edge
        # (cloud→edge). Filled below from the gateway cache.
        "gw_edge_rx": 0, "gw_edge_tx": 0,
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

    if _gateway_cache:
        gw_scope = CGROUP_ROOT / "system.slice" / f"docker-{_gateway_cache['cid']}.scope"
        if gw_scope.exists():
            rx, tx = _read_iface_bytes(_gateway_cache["pid"], _gateway_cache["iface"])
            sums["gw_edge_rx"] = rx
            sums["gw_edge_tx"] = tx
        else:
            # Gateway was restarted — its pid/iface mapping is stale.
            # Mark for refresh; the boundary counter for this snapshot
            # stays at 0, which is fine because the next call's t0/t1
            # delta still cancels out around the gap.
            missing = True

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
        # Directional edge↔cloud bytes read at the toxiproxy gateway's
        # edge-net interface. rx = edge→cloud (coordinator into the cloud
        # subnet), tx = cloud→edge (responses back). No double-count and
        # no intra-cloud inflation — responder↔memory and memory↔storage
        # never traverse this interface.
        "network_edge_to_cloud_bytes": d("gw_edge_rx"),
        "network_cloud_to_edge_bytes": d("gw_edge_tx"),
    }
