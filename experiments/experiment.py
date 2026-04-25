"""Thin client for the coordinator's /experiment endpoint.

POSTs the run config, gets back a JSON list of CSV-ready rows, and upserts
them into results/results.csv. All orchestration (datasets, toxics, agents,
prometheus enrichment, judge calls) lives in the coordinator.

Per-question progress is logged by the coordinator — view via `make logs`
(filter `dmas-coordinator`).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

from lib.run_context import context_fields  # noqa: E402

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://localhost:8010").rstrip("/")
RESULTS_DIR = Path(os.getenv("DMAS_RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))
RESULTS_CSV = RESULTS_DIR / "results.csv"


def _make_run_id() -> str:
    return str(uuid.uuid4())

COLUMNS = [
    "timestamp", "experiment_id", "experiment_name", "experiment_duration_s",
    "seed", "agent1_model", "agent2_model", "agent3_model",
    "memory", "dataset", "conversation_index", "question_id", "question_type",
    "category", "category_label",
    "question", "answer", "gold_answer",
    "f1", "string_sim", "judge_model", "judge_label", "judge_reasoning",
    "toxic_latency", "toxic_jitter", "toxic_bandwidth",
    "peer_threshold_ms", "measured_latency_ms", "peer_help_allowed",
    "peers_asked", "peer_memories", "own_memories",
    "cpu_edge_ns", "cpu_cloud_ns",
    "ram_edge_bytes", "ram_cloud_bytes",
    "disk_edge_bytes", "disk_cloud_bytes",
    "network_edge_bytes", "network_cloud_bytes",
    "agent1_tokens", "agent1_cost_usd",
    "agent1_memory_tokens", "agent1_memory_cost_usd",
    "agent2_tokens", "agent2_cost_usd",
    "agent2_memory_tokens", "agent2_memory_cost_usd",
    "agent3_tokens", "agent3_cost_usd",
    "agent3_memory_tokens", "agent3_memory_cost_usd",
    "total_agent_tokens", "total_agent_cost_usd",
    "total_memory_tokens", "total_memory_cost_usd",
    "total_tokens", "total_cost_usd",
    "max_context_memories", "search_limit",
    "git_sha", "litellm_config_sha", "system_prompt_sha",
    "error",
]

_DEDUPE_KEY = (
    "memory", "dataset", "conversation_index", "seed",
    "agent1_model", "agent2_model", "agent3_model",
    "toxic_latency", "toxic_jitter", "toxic_bandwidth", "peer_threshold_ms",
    "question",
)


_CONFIG_KEY = tuple(k for k in _DEDUPE_KEY if k not in ("seed", "question"))


def _row_key(row: dict) -> tuple:
    return tuple(str(row.get(k, "")) for k in _DEDUPE_KEY)


def _experiment_id(row: dict) -> str:
    """Stable 12-char hash of the configuration columns (dedupe key minus
    `seed` and `question`). All seed×question rows of the same configuration
    share the same id. Use it as the notebook grouping key in place of
    `experiment_name` — the name is human-readable but not unique across runs
    that vary in conversation_index, models, or toxics."""
    raw = "|".join(str(row.get(k, "")) for k in _CONFIG_KEY)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _format_decimals(row: dict) -> None:
    """Force every `_cost_usd` field to a plain decimal string so the CSV
    never contains scientific notation (e.g. `2e-07`). Trailing zeros are
    stripped so `0.0000002000000000` becomes `0.0000002`. Tokens are already
    ints; this only touches float-valued cost columns."""
    for k, v in list(row.items()):
        if not k.endswith("_cost_usd"):
            continue
        if v is None or v == "":
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


def _upsert(rows: list[dict]) -> tuple[int, int]:
    if not rows:
        return (0, 0)
    for r in rows:
        r["experiment_id"] = _experiment_id(r)
        _format_decimals(r)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_keys = {_row_key(r) for r in rows}
    existing: list[dict] = []
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    for r in existing:
        if not r.get("experiment_id"):
            r["experiment_id"] = _experiment_id(r)
        _format_decimals(r)
    kept = [r for r in existing if _row_key(r) not in new_keys]
    replaced = len(existing) - len(kept)
    tmp = RESULTS_CSV.with_suffix(RESULTS_CSV.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            w.writerow(r)
        for r in rows:
            w.writerow(r)
    tmp.replace(RESULTS_CSV)
    return (len(rows) - replaced, replaced)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mem0", "zep", "rag", "none"])
    ap.add_argument("--dataset", required=True, choices=["locomo", "longmemeval"])
    ap.add_argument("--latency", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--bandwidth", type=float, default=0.0)
    ap.add_argument("--peer-threshold-ms", type=float, default=0.0)
    ap.add_argument("--conv", type=int, default=None)
    ap.add_argument("--qid", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model-1", default="gemma4:e4b")
    ap.add_argument("--model-2", default="openai/gpt-4o-mini")
    ap.add_argument("--model-3", default="openai/gpt-4o-mini")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke-test: ask only the first N questions per seed.")
    ap.add_argument("--name-prefix", default="",
                    help="Prefix prepended to experiment_name in CSV rows (e.g. 'test_').")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    body = {
        "backend": args.backend,
        "dataset": args.dataset,
        "conv": args.conv,
        "qid": args.qid,
        "all": args.all,
        "latency": args.latency,
        "jitter": args.jitter,
        "bandwidth": args.bandwidth,
        "peer_threshold_ms": args.peer_threshold_ms,
        "seeds": args.seeds,
        "models": {"1": args.model_1, "2": args.model_2, "3": args.model_3},
        "judge_model": args.judge_model,
        "limit": args.limit,
        "name_prefix": args.name_prefix,
        "context": context_fields(),
    }
    run_id = _make_run_id()
    print(f"[exp] POST {COORDINATOR_URL}/experiment  run_id={run_id}")
    print(f"[exp] coordinator stdout: docker logs -f dmas-coordinator")

    r = requests.post(f"{COORDINATOR_URL}/experiment", json=body, timeout=24 * 3600)
    r.raise_for_status()
    payload = r.json()
    rows: list[dict] = payload.get("rows", [])
    for row in rows:
        row.setdefault("run_id", run_id)
    added, replaced = _upsert(rows)
    print(f"[exp] {added} new + {replaced} replaced row(s) in {RESULTS_CSV}  (n={payload.get('n')})")

    if rows:
        avg_f1 = sum(r["f1"] for r in rows) / len(rows)
        judged = [r for r in rows if r.get("judge_label") is not None]
        judge_msg = ""
        if judged:
            acc = sum(1 for r in judged if r["judge_label"]) / len(judged)
            judge_msg = f"  judge_acc={acc:.4f} (n={len(judged)})"
        print(f"[exp] mean f1={avg_f1:.4f}{judge_msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
