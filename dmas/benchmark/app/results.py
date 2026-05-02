"""Per-(framework, mode) CSV writers.

Each `(name_prefix, memory, mode)` writes to its own file inside
`RESULTS_DIR`, e.g. `test_mem0_unconstrained.csv`. Multiple conversations
for the same backend/mode share a file; the row-level `conversation_index`
column distinguishes them and `experiment_name` carries the conv tag
(`{prefix}{memory}_conv{conv}_{mode}`). `experiment_id` remains a stable
hash over (memory, conv, name_prefix, toxics) so per-conv resume keying
stays unambiguous. Every /experiment call reads the matching file,
filters by (conv, ...), and skips question entries already present.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))

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
    #   responder_context_tokens — `prompt_tokens` of the OpenAI
    #     completion that produced the final answer; the actual
    #     context length the answering LLM consumed for this question.
    #     Used in efficiency analyses to normalise accuracy by how
    #     much retrieved context the responder had to process.
    "coordinator_asked_responder", "memories_returned", "top_k", "search_calls",
    "responder_context_tokens",
    # LLM-as-judge consensus columns. The `judge` column carries the
    # majority-vote final verdict; the columns below preserve the raw
    # ballots so analyses can compute inter-judge agreement.
    #   judge_n             — number of independent judge calls per row
    #                         (LLM_AS_JUDGE_SEED on the request).
    #   judge_correct_votes — count of those calls that returned CORRECT.
    #   judge_labels        — pipe-separated list of every individual
    #                         label, in call order (e.g. "CORRECT|WRONG|CORRECT").
    "judge_n", "judge_correct_votes", "judge_labels",
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


def _is_constrained(latency: Any, jitter: Any, bandwidth: Any) -> bool:
    def _f(v: Any) -> float:
        try:
            return float(v) if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0
    return _f(latency) > 0 or _f(jitter) > 0 or _f(bandwidth) > 0


def _slug(memory: Any, name_prefix: Any,
          latency: Any, jitter: Any, bandwidth: Any) -> str:
    mode = "constrained" if _is_constrained(latency, jitter, bandwidth) else "unconstrained"
    prefix = "" if name_prefix in (None, "") else str(name_prefix)
    return f"{prefix}{memory}_{mode}"


def file_for_config(memory: str, latency: float, jitter: float,
                    bandwidth: float, name_prefix: str = "") -> Path:
    return RESULTS_DIR / f"{_slug(memory, name_prefix, latency, jitter, bandwidth)}.csv"


def _file_for_row(row: dict[str, Any]) -> Path:
    return RESULTS_DIR / f"{_slug(row.get('memory', ''), row.get('name_prefix', ''), row.get('toxic_latency', 0), row.get('toxic_jitter', 0), row.get('toxic_bandwidth', 0))}.csv"


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
    path = _file_for_row(row)
    is_empty = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if is_empty:
            w.writeheader()
        w.writerow(row)


def already_done(memory: str, conv_index: int, latency: float, jitter: float,
                 bandwidth: float, name_prefix: str = "") -> set[str]:
    """Return {question} text already present as ask-phase rows for this
    `(memory, conv, mode, prefix)` configuration. The bench now runs each
    question once (judging is repeated, not asking), so resume keys on
    `question` alone — `(seed, question)` no longer applies. Reads the
    `(memory, mode, prefix)` file and filters rows by `conversation_index`
    so multiple convs sharing the file don't collide.
    """
    path = file_for_config(memory, latency, jitter, bandwidth, name_prefix)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("phase") == "ask" and r.get("conversation_index", "") == str(conv_index):
                q = r.get("question") or ""
                if q:
                    done.add(q)
    return done


def loaded_messages(memory: str, conv_index: int, latency: float, jitter: float,
                    bandwidth: float, name_prefix: str = "") -> set[str]:
    """Return {message-counter} markers already saved for this config.

    Resume marker: each persisted message writes one row whose `question`
    is the global counter (1, 2, ..., N) within the conv. `category` holds
    the session number. The bench skips any counter already in this set —
    granularity is per-message because each call to coordinator/memorize
    carries exactly one message. Filters by `conversation_index` since
    the file now holds rows from every conv for the same backend/mode.
    """
    path = file_for_config(memory, latency, jitter, bandwidth, name_prefix)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("phase") == "load" and r.get("conversation_index", "") == str(conv_index):
                q = r.get("question") or ""
                if q:
                    # Any prior attempt counts — including `skipped` (empty
                    # turns the backend filtered out) and `failed`. Without
                    # this, those rows would be re-POSTed on every resume
                    # and pile up. To retry a failed row, delete it from
                    # the per-experiment CSV first.
                    done.add(q)
    return done
