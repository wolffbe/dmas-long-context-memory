"""Direct read of litellm's `/metrics` Prometheus exposition endpoint.

Returns counter values at the moment of the GET — no scrape lag, no
TSDB rounding. Two snapshots before+after a single LLM call attribute
true per-call tokens and cost.

Output is split by *deployment group*: edge (local, ollama-served) vs
cloud (OpenAI-served). Classification is by `model` label on each
litellm metric line, matched against the comma-separated set in
EDGE_MODELS env (defaults to OLLAMA_MODEL — the local model). Anything
else is cloud.
"""
from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

# Comma-separated list of /metrics endpoints. With the edge/cloud split
# there are two litellm instances; each exposes its own counters and
# they're summed at read time. Falls back to LITELLM_METRICS_URL
# (singular) for any service still wired to a single endpoint, and to
# the cloud default as a last resort.
_RAW_URLS = (
    os.getenv("LITELLM_METRICS_URLS")
    or os.getenv("LITELLM_METRICS_URL")
    or "http://litellm-cloud:4000/metrics"
)
LITELLM_METRICS_URLS: tuple[str, ...] = tuple(
    u.strip() for u in _RAW_URLS.split(",") if u.strip()
)


def _edge_models() -> set[str]:
    """Models LiteLLM serves locally (Ollama). Defaults to the single
    OLLAMA_MODEL tag — both `model_name` and `model` in the templated
    litellm config substitute to it, so no separate alias is needed."""
    raw = os.getenv("EDGE_MODELS") or os.getenv("OLLAMA_MODEL", "")
    return {m.strip() for m in raw.split(",") if m.strip()}


_LINE = re.compile(
    r'^(litellm_total_tokens_metric_total|litellm_spend_metric_total)\{([^}]*)\}\s+([\d.eE+-]+)\s*$',
    re.MULTILINE,
)
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _parse(text: str) -> dict[str, float]:
    out = {"edge_tokens": 0.0, "edge_cost": 0.0,
           "cloud_tokens": 0.0, "cloud_cost": 0.0}
    edge_models = _edge_models()
    for m in _LINE.finditer(text):
        name = m.group(1)
        labels = {lm.group(1): lm.group(2) for lm in _LABEL.finditer(m.group(2))}
        model = labels.get("model") or labels.get("requested_model") or ""
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        group = "edge" if model in edge_models else "cloud"
        if name == "litellm_total_tokens_metric_total":
            out[f"{group}_tokens"] += v
        else:
            out[f"{group}_cost"] += v
    out["edge_tokens"] = int(out["edge_tokens"])
    out["cloud_tokens"] = int(out["cloud_tokens"])
    return out


# Sync client for the memory service path. Lazy/module-level so the TCP
# connection to litellm is reused across snapshots.
_sync_client = httpx.Client(timeout=2.0, follow_redirects=True)


def _zero() -> dict[str, float]:
    return {"edge_tokens": 0.0, "edge_cost": 0.0,
            "cloud_tokens": 0.0, "cloud_cost": 0.0}


def _accumulate(acc: dict[str, float], parsed: dict[str, float]) -> None:
    for k, v in parsed.items():
        if k.endswith("_tokens"):
            acc[k] = int(acc.get(k, 0)) + int(v)
        else:
            acc[k] = float(acc.get(k, 0.0)) + float(v)


def usage_snapshot_sync() -> dict[str, float]:
    """Return per-group totals at the current instant (sync), summed
    across every endpoint in LITELLM_METRICS_URLS.

    Failures on any single endpoint degrade to all-zero for that one
    and the others still contribute — partial reads never mis-attribute
    a delta later because both t0 and t1 see the same set of failures.
    """
    total = _zero()
    for url in LITELLM_METRICS_URLS:
        try:
            text = _sync_client.get(url).text
        except Exception as exc:
            logger.debug("litellm /metrics fetch failed (%s): %s", url, exc)
            continue
        _accumulate(total, _parse(text))
    return total


async def usage_snapshot_async(client: httpx.AsyncClient) -> dict[str, float]:
    """Async equivalent of `usage_snapshot_sync`. Caller-supplied client
    so connection pooling stays under the caller's control."""
    total = _zero()
    for url in LITELLM_METRICS_URLS:
        try:
            r = await client.get(url, timeout=2.0, follow_redirects=True)
            text = r.text
        except Exception as exc:
            logger.debug("litellm /metrics fetch failed (%s): %s", url, exc)
            continue
        _accumulate(total, _parse(text))
    return total


def diff(t0: dict[str, float], t1: dict[str, float]) -> dict[str, float]:
    return {
        "edge_tokens":  max(0, int(t1.get("edge_tokens", 0)) - int(t0.get("edge_tokens", 0))),
        "edge_cost":    max(0.0, float(t1.get("edge_cost", 0)) - float(t0.get("edge_cost", 0))),
        "cloud_tokens": max(0, int(t1.get("cloud_tokens", 0)) - int(t0.get("cloud_tokens", 0))),
        "cloud_cost":   max(0.0, float(t1.get("cloud_cost", 0)) - float(t0.get("cloud_cost", 0))),
    }
