"""Manage toxiproxy proxies and toxics for the experiment harness.

The benchmark owns the toxic state — operators just pick latency/jitter/
bandwidth on each `/experiment` call and the service syncs
toxiproxy to match. `apply_all` is idempotent: it ensures the expected
proxies exist (memory, responder), removes any toxics not in the desired
set, adds the ones that are missing, and updates ones that drifted.
`verify_all` re-reads state and raises HTTP 412 on any mismatch so a
mid-run drift still aborts the experiment instead of silently mis-recording.
"""
from __future__ import annotations

from typing import Any, Iterable

import httpx
from fastapi import HTTPException


PROXY_SPECS: tuple[tuple[str, str, str], ...] = (
    # (name, listen, upstream) — listen is on toxiproxy:1800X, upstream is the
    # internal service host:port on dmas-network.
    ("memory", "0.0.0.0:18005", "memory:8002"),
    ("responder", "0.0.0.0:18006", "responder:8003"),
)


def _expected(latency: float, jitter: float, bandwidth: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if latency > 0 or jitter > 0:
        out["latency"] = {
            "type": "latency",
            "stream": "downstream",
            "attributes": {"latency": int(latency), "jitter": int(jitter)},
        }
    if bandwidth > 0:
        out["bandwidth"] = {
            "type": "bandwidth",
            "stream": "downstream",
            "attributes": {"rate": int(bandwidth)},
        }
    return out


async def _proxies(client: httpx.AsyncClient, admin: str) -> dict[str, dict[str, Any]]:
    r = await client.get(f"{admin}/proxies")
    r.raise_for_status()
    return r.json()


async def _toxics_on(client: httpx.AsyncClient, admin: str, proxy: str) -> dict[str, dict[str, Any]]:
    r = await client.get(f"{admin}/proxies/{proxy}/toxics")
    r.raise_for_status()
    return {t["name"]: t for t in r.json()}


def _matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        actual.get("type") == expected["type"]
        and actual.get("stream") == expected["stream"]
        and actual.get("attributes") == expected["attributes"]
    )


async def _ensure_proxies(client: httpx.AsyncClient, admin: str) -> None:
    """Create any of PROXY_SPECS that are missing on this admin."""
    existing = await _proxies(client, admin)
    for name, listen, upstream in PROXY_SPECS:
        if name in existing:
            continue
        body = {"name": name, "listen": listen, "upstream": upstream, "enabled": True}
        r = await client.post(f"{admin}/proxies", json=body)
        if r.status_code >= 400:
            raise HTTPException(502, f"toxiproxy {admin}: failed to create proxy {name!r}: {r.status_code} {r.text[:200]}")


async def apply_all(
    client: httpx.AsyncClient,
    admins: Iterable[str],
    latency: float,
    jitter: float,
    bandwidth: float,
) -> list[dict[str, Any]]:
    """Sync every admin to the expected toxic set. Idempotent."""
    expected = _expected(latency, jitter, bandwidth)
    report: list[dict[str, Any]] = []
    for admin in admins:
        try:
            await _ensure_proxies(client, admin)
            present = await _proxies(client, admin)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"toxiproxy admin {admin} unreachable: {exc}")
        for proxy in present:
            actual = await _toxics_on(client, admin, proxy)
            # Drop any toxics we don't want or that drifted from expected.
            for name, cur in actual.items():
                if name not in expected or not _matches(cur, expected[name]):
                    dr = await client.delete(f"{admin}/proxies/{proxy}/toxics/{name}")
                    if dr.status_code >= 400 and dr.status_code != 404:
                        raise HTTPException(502, f"toxiproxy {admin}/{proxy}: delete {name} failed: {dr.status_code} {dr.text[:200]}")
            # Re-add anything missing.
            after = await _toxics_on(client, admin, proxy)
            for name, want in expected.items():
                if name in after and _matches(after[name], want):
                    continue
                body = {"name": name, **want}
                cr = await client.post(f"{admin}/proxies/{proxy}/toxics", json=body)
                if cr.status_code >= 400:
                    raise HTTPException(502, f"toxiproxy {admin}/{proxy}: create {name} failed: {cr.status_code} {cr.text[:200]}")
            report.append({"admin": admin, "proxy": proxy, "applied": sorted(expected)})
    return report


async def verify_all(
    client: httpx.AsyncClient,
    admins: Iterable[str],
    latency: float,
    jitter: float,
    bandwidth: float,
) -> list[dict[str, Any]]:
    expected = _expected(latency, jitter, bandwidth)
    report: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for admin in admins:
        try:
            proxies = await _proxies(client, admin)
        except Exception as exc:
            raise HTTPException(502, f"toxiproxy admin {admin} unreachable: {exc}")
        for proxy in proxies:
            actual = await _toxics_on(client, admin, proxy)
            extra = set(actual) - set(expected)
            missing = set(expected) - set(actual)
            mismatched = [
                n for n in expected.keys() & actual.keys()
                if not _matches(actual[n], expected[n])
            ]
            ok = not (extra or missing or mismatched)
            entry = {
                "admin": admin, "proxy": proxy, "ok": ok,
                "missing": sorted(missing), "extra": sorted(extra),
                "mismatched": sorted(mismatched),
            }
            report.append(entry)
            if not ok:
                mismatches.append(f"{admin}/{proxy}: missing={sorted(missing)} extra={sorted(extra)} mismatched={sorted(mismatched)}")
    if mismatches:
        raise HTTPException(412, "toxic drift detected: " + " | ".join(mismatches))
    return report
