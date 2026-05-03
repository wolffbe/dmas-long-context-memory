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

LITELLM_METRICS_URL = os.getenv(
    "LITELLM_METRICS_URL",
    "http://litellm:4000/metrics",
)


def _edge_models() -> set[str]:
    """Models LiteLLM serves locally (Ollama). Includes both the LiteLLM
    alias the coordinator actually requests (`local-slm`) and the raw
    Ollama tag — LiteLLM's metric labels can carry either depending on
    which one wins the labeling race in `requested_model` vs `model`."""
    raw = os.getenv("EDGE_MODELS") or os.getenv("OLLAMA_MODEL", "")
    models = {m.strip() for m in raw.split(",") if m.strip()}
    models.add("local-slm")
    return models


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


def usage_snapshot_sync() -> dict[str, float]:
    """Return per-group totals at the current instant (sync).

    Failures degrade to all-zero so a flaky read never mis-attributes
    a delta later.
    """
    try:
        text = _sync_client.get(LITELLM_METRICS_URL).text
    except Exception as exc:
        logger.debug("litellm /metrics fetch failed: %s", exc)
        return {"edge_tokens": 0.0, "edge_cost": 0.0,
                "cloud_tokens": 0.0, "cloud_cost": 0.0}
    return _parse(text)


async def usage_snapshot_async(client: httpx.AsyncClient) -> dict[str, float]:
    """Async equivalent of `usage_snapshot_sync`. Caller-supplied client
    so connection pooling stays under the caller's control."""
    try:
        r = await client.get(LITELLM_METRICS_URL, timeout=2.0, follow_redirects=True)
        text = r.text
    except Exception as exc:
        logger.debug("litellm /metrics fetch failed: %s", exc)
        return {"edge_tokens": 0.0, "edge_cost": 0.0,
                "cloud_tokens": 0.0, "cloud_cost": 0.0}
    return _parse(text)


def diff(t0: dict[str, float], t1: dict[str, float]) -> dict[str, float]:
    return {
        "edge_tokens":  max(0, int(t1.get("edge_tokens", 0)) - int(t0.get("edge_tokens", 0))),
        "edge_cost":    max(0.0, float(t1.get("edge_cost", 0)) - float(t0.get("edge_cost", 0))),
        "cloud_tokens": max(0, int(t1.get("cloud_tokens", 0)) - int(t0.get("cloud_tokens", 0))),
        "cloud_cost":   max(0.0, float(t1.get("cloud_cost", 0)) - float(t0.get("cloud_cost", 0))),
    }
