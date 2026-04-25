"""Sync toxiproxy toxics across multiple admin endpoints.

A `Toxic` payload mirrors the toxiproxy admin API shape:
    {"name": str?, "type": str, "stream": "upstream"|"downstream"?,
     "attributes": dict, "proxies": list[str]?}

If `name` is omitted, `type` is used. If `proxies` is omitted, the toxic is
applied to every proxy registered on each admin. The sync is idempotent: an
existing toxic with matching type/stream/attributes is left alone; one with
the same name but different fields is replaced; toxics whose names no longer
appear in `desired` are deleted.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx


def _toxic_name(t: dict[str, Any]) -> str:
    name = t.get("name") or t.get("type")
    if not name:
        raise ValueError(f"toxic missing both name and type: {t!r}")
    return str(name)


def _toxic_payload(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _toxic_name(t),
        "type": t["type"],
        "stream": t.get("stream", "downstream"),
        "attributes": t.get("attributes", {}),
        "toxicity": float(t.get("toxicity", 1.0)),
    }


def _matches(existing: dict[str, Any], payload: dict[str, Any]) -> bool:
    return (
        existing.get("type") == payload["type"]
        and existing.get("stream") == payload["stream"]
        and existing.get("attributes") == payload["attributes"]
        and float(existing.get("toxicity", 1.0)) == payload["toxicity"]
    )


async def _sync_proxy(
    client: httpx.AsyncClient,
    admin: str,
    proxy_name: str,
    desired: list[dict[str, Any]],
) -> dict[str, Any]:
    base = f"{admin}/proxies/{proxy_name}/toxics"
    r = await client.get(base)
    r.raise_for_status()
    existing = {t["name"]: t for t in r.json()}
    desired_by_name = {_toxic_name(t): _toxic_payload(t) for t in desired}

    actions: list[str] = []
    for name in list(existing):
        if name not in desired_by_name:
            d = await client.delete(f"{base}/{name}")
            d.raise_for_status()
            actions.append(f"-{name}")

    for name, payload in desired_by_name.items():
        cur = existing.get(name)
        if cur and _matches(cur, payload):
            actions.append(f"={name}")
            continue
        if cur:
            d = await client.delete(f"{base}/{name}")
            d.raise_for_status()
        c = await client.post(base, json=payload)
        c.raise_for_status()
        actions.append(f"+{name}")

    return {"proxy": proxy_name, "actions": actions}


async def _sync_admin(
    client: httpx.AsyncClient,
    admin: str,
    desired: list[dict[str, Any]],
    targets: set[str] | None,
) -> dict[str, Any]:
    r = await client.get(f"{admin}/proxies")
    r.raise_for_status()
    available = list(r.json().keys())
    proxy_names = [p for p in available if targets is None or p in targets]
    results = await asyncio.gather(
        *(_sync_proxy(client, admin, p, desired) for p in proxy_names),
        return_exceptions=True,
    )
    out: list[dict[str, Any]] = []
    for p, res in zip(proxy_names, results):
        if isinstance(res, Exception):
            out.append({"proxy": p, "error": f"{type(res).__name__}: {res}"})
        else:
            out.append(res)
    return {"admin": admin, "proxies": out}


async def sync_all(
    client: httpx.AsyncClient,
    admins: tuple[str, ...],
    desired: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply `desired` toxics to every (admin, proxy) pair.

    Per-toxic `proxies` lists are honored: a toxic with `"proxies": ["peers"]`
    is only applied where a proxy named "peers" exists, and toxics without
    that field hit every proxy.
    """
    if not admins:
        return []

    # Build the union of proxy targets requested across all toxics; if any
    # toxic omits `proxies`, we apply to every proxy on each admin.
    targets: set[str] | None = set()
    for t in desired:
        ps = t.get("proxies")
        if ps is None:
            targets = None
            break
        targets.update(ps)

    tasks = [_sync_admin(client, a, desired, targets) for a in admins]
    return list(await asyncio.gather(*tasks))
