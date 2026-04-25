"""Single CSV writer for every experiment row.

`experiment_id` is a stable hash over the configuration columns (memory,
conv, models, toxics) so all (seed, question) rows of one configuration
share an id. Resume-by-config: every /experiment call reads existing rows
for the matching configuration and skips (seed, question) pairs already
present.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))
RESULTS_CSV = RESULTS_DIR / "results.csv"

COLUMNS = [
    "timestamp", "experiment_id", "experiment_name", "phase",
    "memory", "conversation_index", "name_prefix",
    "seed", "question", "answer", "gold_answer",
    "category",
    "judge", "judge_reason",
    "toxic_latency", "toxic_jitter", "toxic_bandwidth",
    # wall_ms = compute_ms + flush_ms.
    #   compute_ms is the user-facing request latency (t_call_start →
    #     HTTP response returned); this is what a production system pays.
    #   flush_ms is the bench-side wait we add AFTER the response so
    #     async DB checkpoints (Neo4j) land in this row's
    #     disk_cloud_bytes — instrumentation artifact, not real cost.
    "wall_ms", "compute_ms", "flush_ms",
    "cpu_edge_ns", "cpu_cloud_ns",
    "ram_edge_peak_bytes", "ram_cloud_peak_bytes",
    "disk_edge_bytes", "disk_cloud_bytes",
    "network_edge_bytes", "network_cloud_bytes",
    # LLM tokens/cost split by where the model is served:
    #   edge_llm_* — local ollama (free in litellm pricing).
    #   cloud_llm_* — OpenAI passthrough.
    #   llm_*      — sum of the two; cached for analysis convenience.
    "edge_llm_tokens", "edge_llm_cost_usd",
    "cloud_llm_tokens", "cloud_llm_cost_usd",
    "llm_tokens", "llm_cost_usd",
    # Per-ask retrieval stats. Only set on `phase=ask` rows; load/warmup
    # rows leave these blank.
    #   coordinator_asked_responder — True if the SLM emitted the
    #     ask_responder tool call, False if the fallback path was used
    #     (qwen2.5:3b ignored tool_choice="required").
    #   memories_returned — sum of memory items the responder received
    #     across all search_memories tool calls for this question.
    #   top_k — backend's configured retrieval ceiling (env
    #     MEMORIES_SEARCH_LIMIT). Null for full_context which dumps the
    #     whole conversation regardless of k.
    #   search_calls — number of search_memories invocations the
    #     responder made (typically 1).
    "coordinator_asked_responder", "memories_returned", "top_k", "search_calls",
    "error",
]

_DEDUPE_KEY = (
    "memory", "conversation_index", "name_prefix",
    "toxic_latency", "toxic_jitter", "toxic_bandwidth",
    "seed", "question",
)
_CONFIG_KEY = tuple(k for k in _DEDUPE_KEY if k not in ("seed", "question"))


def experiment_id(row: dict[str, Any]) -> str:
    raw = "|".join(str(row.get(k, "")) for k in _CONFIG_KEY)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _format_decimals(row: dict[str, Any]) -> None:
    for k, v in list(row.items()):
        if not k.endswith("_cost_usd") or v in (None, ""):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == 0.0:
            row[k] = "0.0"
            continue
        s = f"{f:.10f}".rstrip("0")
        row[k] = s + "0" if s.endswith(".") else s


def append_row(row: dict[str, Any]) -> None:
    row["experiment_id"] = experiment_id(row)
    _format_decimals(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_empty = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if is_empty:
            w.writeheader()
        w.writerow(row)


def _matches_config(r: dict[str, Any], memory: str, conv_index: int,
                    latency: float, jitter: float, bandwidth: float,
                    name_prefix: str = "") -> bool:
    return (
        r.get("memory", "") == str(memory)
        and r.get("conversation_index", "") == str(conv_index)
        and r.get("name_prefix", "") == str(name_prefix)
        and r.get("toxic_latency", "") == str(latency)
        and r.get("toxic_jitter", "") == str(jitter)
        and r.get("toxic_bandwidth", "") == str(bandwidth)
    )


def already_done(memory: str, conv_index: int, latency: float, jitter: float,
                 bandwidth: float, name_prefix: str = "") -> set[tuple[int, str]]:
    """Return {(seed, question)} pairs already present as ask-phase rows."""
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return set()
    done: set[tuple[int, str]] = set()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("phase") == "ask" and _matches_config(r, memory, conv_index, latency, jitter, bandwidth, name_prefix):
                try:
                    done.add((int(r["seed"]), r["question"]))
                except (KeyError, ValueError, TypeError):
                    continue
    return done


def loaded_messages(memory: str, conv_index: int, latency: float, jitter: float,
                    bandwidth: float, name_prefix: str = "") -> set[str]:
    """Return {message-counter} markers already saved for this config.

    Resume marker: each persisted message writes one row whose `question`
    is the global counter (1, 2, ..., N) within the conv. `category` holds
    the session number. The bench skips any counter already in this set —
    granularity is per-message because each call to coordinator/memorize
    carries exactly one message.
    """
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("phase") == "load" and _matches_config(r, memory, conv_index, latency, jitter, bandwidth, name_prefix):
                q = r.get("question") or ""
                if q:
                    # Any prior attempt counts — including `skipped` (empty
                    # turns the backend filtered out) and `failed`. Without
                    # this, those rows would be re-POSTed on every resume
                    # and pile up. To retry a failed row, delete it from
                    # results.csv first.
                    done.add(q)
    return done
