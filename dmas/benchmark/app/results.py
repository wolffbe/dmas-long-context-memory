"""Per-(framework, mode) CSV writers.

Each `(name_prefix, memory, mode)` writes to its own file inside
`RESULTS_DIR`, e.g. `test_mem0_unconstrained.csv`. Multiple conversations
for the same backend/mode share a file; the row-level `conversation_index`
column distinguishes them and `experiment_name` carries the conv tag
(`{prefix}{memory}_conv{conv}_{mode}`). Every /experiment call reads the
matching file, filters by (conv, ...), and skips question entries
already present.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/results"))

COLUMNS = [
    # Front-loaded identifiers so a quick `head` on the CSV shows the
    # experiment + when it ran before any of the noisy id/trace columns.
    "experiment_name", "timestamp",
    # Langfuse session id — `{experiment_name}_{run_id}`, where `run_id`
    # is generated once per /experiment call so two runs of the same
    # config land in two separate Langfuse sessions. Propagated to every
    # span via the `langfuse.session.id` OTel attribute. Notebook builds
    # the session URL as
    # `${LANGFUSE_PUBLIC_URL}/project/dmas/sessions/{session_id}`.
    "session_id",
    # 32-hex OTel trace_id of the span that produced this row
    # (warmup → `warmup`; load → `load.message`; ask → `ask.question`).
    # Each row's primary span is its own detached-root trace, so this is
    # the per-trace key. Use it to deep-link from a CSV row to the
    # exact trace in Langfuse.
    "trace_id",
    # Ask rows only — trace_id of the `judge.evaluate` span (detached
    # root, separate trace from `ask.question`). Lets the notebook deep-
    # link from a CSV row to the judge's reasoning trace.
    "judge_trace_id",
    "phase",
    "memory", "mode", "conversation_index",
    # Per-row resume aides (informational; resume key is still
    # `question` which carries the global counter on load rows and the
    # question text on ask rows). turn_index is the 0-based msg index
    # within the session (set on load rows, blank on ask). question_index
    # is the 1-based ordinal within the filtered question set for this
    # leg (set on ask rows, blank on load).
    "turn_index", "question_index",
    # Per-ask retrieval stats. Only set on `phase=ask` rows; load/warmup
    # rows leave these blank.
    #   memories_returned — sum of memory items the responder received
    #     across all search_memories tool calls for this question.
    #   top_k — backend's configured retrieval ceiling (env
    #     MEMORIES_SEARCH_LIMIT). Null for full_context which dumps the
    #     whole conversation regardless of k.
    #   search_calls — number of search_memories invocations the
    #     responder made (typically 1).
    "search_calls", "top_k", "memories_returned",
    "question_category", "question", "answer", "gold_answer",
    "seed",
    # LLM-as-judge consensus.
    #   judge_n       — number of independent judge calls per row
    #                   (request's `llm_as_judge_seed`).
    #   judge_verdict — final majority label (CORRECT iff strictly more
    #                   than half the calls returned CORRECT, else WRONG;
    #                   PLACEHOLDER/ERROR pass through if every call agrees).
    "judge_n", "judge_verdict", "judge_reason",
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
    # Directional edge↔cloud bytes, read at the toxiproxy gateway's
    # edge-net veth. No intra-cloud or storage traffic is included by
    # construction — see cgroup_metrics.delta() for the wiring.
    "network_edge_to_cloud_bytes", "network_cloud_to_edge_bytes",
    # LLM tokens/cost split by where the model is served:
    #   edge_llm_* — local ollama (free in litellm pricing).
    #   cloud_llm_* — OpenAI passthrough.
    #   llm_*      — sum of the two; cached for analysis convenience.
    "edge_llm_tokens", "edge_llm_cost_usd",
    "cloud_llm_tokens", "cloud_llm_cost_usd",
    "llm_tokens", "llm_cost_usd",
    # `prompt_tokens` of the OpenAI completion that produced the final
    # answer — the actual context length the responder LLM consumed for
    # this question. These columns are a SLICE of cloud_llm_tokens /
    # cloud_llm_cost_usd (already counted there), tracked separately so
    # efficiency analyses can normalise accuracy by how much retrieved
    # context the responder had to process. Do NOT add them to llm_*
    # totals — they're already included via cloud_*. The cost column
    # prices the prompt slice through litellm's cost table for the
    # responder model (rate stays in lockstep with the proxy config).
    "responder_context_window_tokens", "responder_context_window_cost_usd",
    # Total milliseconds the responder spent sleeping inside its
    # `_chat_with_retry` helper to ride out OpenAI 429s on this row.
    # The sleep duration is read from the failing response's
    # `x-ratelimit-reset-{requests,tokens}` headers, so this is the
    # exact throttle window OpenAI told us to wait. Not subtracted
    # from compute_ms/wall_ms at write time — analysis subtracts it
    # to get `effective_compute_ms` (the throttle-free view) while the
    # raw columns preserve what the user actually waited. 0 on rows
    # with no rate-limit retries (the typical case at Tier 3+).
    #
    # NOTE: the cpu/disk/network/ram counters above are NOT adjusted
    # for retry-wait. During sleep the responder consumes ~0 of each
    # (the OpenAI socket is closed; time.sleep blocks the thread), so
    # its contribution to those cgroup counters is ~0 already. The only
    # contamination is background work from other "cloud" cgroup-mates
    # (langfuse-web, neo4j checkpoints, OTel heartbeats) during long
    # retries — small enough that the raw counters remain accurate
    # to within the noise floor of the measurement.
    "cloud_llm_retry_wait_ms",
    "error",
]

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
    _format_decimals(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _file_for_row(row)
    is_empty = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if is_empty:
            w.writeheader()
        w.writerow(row)


def _iter_phase_rows(memory: str, conv_index: int, latency: float, jitter: float,
                     bandwidth: float, name_prefix: str, phase: str,
                     session_id: str | None):
    """Yield CSV rows for the per-(memory, mode, prefix) file, filtered to
    the requested phase + conversation_index (and optionally session_id).
    Shared by `already_done` (phase=ask) and `loaded_messages` (phase=load)."""
    path = file_for_config(memory, latency, jitter, bandwidth, name_prefix)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("phase") != phase or r.get("conversation_index", "") != str(conv_index):
                continue
            if session_id is not None and r.get("session_id", "") != session_id:
                continue
            yield r


def already_done(memory: str, conv_index: int, latency: float, jitter: float,
                 bandwidth: float, name_prefix: str = "",
                 session_id: str | None = None) -> set[str]:
    """Return {question} text already present as ask-phase rows for this
    `(memory, conv, mode, prefix)` configuration. The bench now runs each
    question once (judging is repeated, not asking), so resume keys on
    `question` alone — `(seed, question)` no longer applies. Reads the
    `(memory, mode, prefix)` file and filters rows by `conversation_index`
    so multiple convs sharing the file don't collide.

    When `session_id` is given, only rows tagged with the same session_id
    are treated as resumable. Rows from prior runs (different session_id)
    are NOT skipped, so a fresh /experiment call always re-asks even if
    the (memory, conv, mode, prefix) file already has answers from an
    earlier session. Pass session_id explicitly on /experiment to resume
    a specific run.
    """
    done: set[str] = set()
    for r in _iter_phase_rows(memory, conv_index, latency, jitter, bandwidth,
                              name_prefix, phase="ask", session_id=session_id):
        q = r.get("question") or ""
        if q:
            done.add(q)
    return done


def count_conv_rows(memory: str, conv_index: int, latency: float, jitter: float,
                    bandwidth: float, name_prefix: str = "") -> tuple[int, int]:
    """Return `(n_load, n_ask)` — count of load/ask rows for this conv in
    the per-(memory, mode, prefix) CSV. Used by /experiment to decide
    whether a leg is already complete vs. needs a fresh run.
    Session-agnostic: any prior run's rows count.
    """
    n_load = n_ask = 0
    for r in _iter_phase_rows(memory, conv_index, latency, jitter, bandwidth,
                              name_prefix, phase="load", session_id=None):
        n_load += 1
    for r in _iter_phase_rows(memory, conv_index, latency, jitter, bandwidth,
                              name_prefix, phase="ask", session_id=None):
        n_ask += 1
    return n_load, n_ask


def purge_conv(memory: str, conv_index: int, latency: float, jitter: float,
               bandwidth: float, name_prefix: str = "") -> int:
    """Remove every row for `conv_index` from the per-(memory, mode, prefix)
    CSV. Other convs sharing the file are preserved. Returns the number of
    rows removed. Writes via tmpfile + atomic rename so a crash mid-purge
    leaves the original file intact.
    """
    path = file_for_config(memory, latency, jitter, bandwidth, name_prefix)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    removed = 0
    with path.open("r", newline="", encoding="utf-8") as src, \
         tmp.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in reader:
            if r.get("conversation_index", "") == str(conv_index):
                removed += 1
                continue
            writer.writerow(r)
    os.replace(tmp, path)
    return removed


def loaded_messages(memory: str, conv_index: int, latency: float, jitter: float,
                    bandwidth: float, name_prefix: str = "",
                    session_id: str | None = None) -> set[str]:
    """Return {message-counter} markers already saved for this config.

    Resume marker: each persisted message writes one row whose `question`
    is the global counter (1, 2, ..., N) within the conv. `category` holds
    the session number. The bench skips any counter already in this set —
    granularity is per-message because each call to coordinator/memorize
    carries exactly one message. Filters by `conversation_index` since
    the file now holds rows from every conv for the same backend/mode.

    When `session_id` is given, only rows tagged with the same session_id
    count toward resume — load rows from prior sessions don't suppress a
    fresh load. This pairs with `already_done` so a new /experiment call
    (no `session_id` passed) starts clean instead of inheriting the
    previous run's CSV state.
    """
    done: set[str] = set()
    # Any prior attempt counts — including `skipped` (empty turns the
    # backend filtered out) and `failed`. Without this, those rows would
    # be re-POSTed on every resume and pile up. To retry a failed row,
    # delete it from the per-experiment CSV first.
    for r in _iter_phase_rows(memory, conv_index, latency, jitter, bandwidth,
                              name_prefix, phase="load", session_id=session_id):
        q = r.get("question") or ""
        if q:
            done.add(q)
    return done
