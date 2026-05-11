"""Benchmark service HTTP API.

`/experiment` is the single operator-facing entrypoint. Per call:
  for each backend in `backends`:
    - reset memory state (drop qdrant collections / wipe Neo4j graph)
    - sync toxiproxy to the requested mode (constrained|unconstrained)
    - load the requested LOCOMO conversation through the coordinator
    - ask every (non-adversarial) question once; the LLM-as-judge runs
      `llm_as_judge_seed` times and majority-votes the verdict
    - reset memory state again so the next backend / next call is clean

Each ask judges the answer with the LLM-as-a-judge grader, measures the
resource delta directly from /sys/fs/cgroup (zero-cache) and litellm's
/metrics endpoint, and appends one row per question to a per-(framework,
mode) CSV under /results/ named {prefix}{memory}_{mode}.csv. Multiple
convs share the file and are distinguished by `conversation_index` and
`experiment_name`. Resume is per-question: rows already present for the
exact (backend, conv, mode, question) are skipped. NDJSON progress is
streamed back to the caller.
"""
from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from shared import otel_init

otel_init.init("benchmark")
otel_init.instrument_httpx()

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from opentelemetry import trace as otel_trace
from opentelemetry.context import Context as OTelContext
from opentelemetry.trace import format_trace_id

from app.agents import post_ask, post_memorize, post_reset, post_warmup
from app.judges import judge_majority
from app.locomo_service import LocomoService
from app.results import RESULTS_DIR, already_done, append_row, loaded_messages
from shared.litellm_usage import usage_snapshot_async as litellm_usage_snapshot, diff as litellm_usage_diff
from app.cgroup_metrics import snapshot as cgroup_snapshot, delta as cgroup_delta, wait_io_quiet as cgroup_wait_io_quiet
from app.toxics import apply_all, verify_all

_tracer = otel_trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bench] %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

TOXIPROXY_ADMINS: tuple[str, ...] = tuple(
    a.strip() for a in os.getenv("TOXIPROXY_ADMINS", "http://toxiproxy:8474").split(",") if a.strip()
)

VALID_BACKENDS = ("mem0", "graphiti", "rag", "cognee", "full_context")

# Constrained-mode network profile. Single source of truth: the operator
# picks the *mode*, the benchmark picks the toxic values. Override per
# deployment via env if a different regime is desired.
CONSTRAINED_LATENCY = float(os.getenv("CONSTRAINED_LATENCY_MS", "150"))
CONSTRAINED_JITTER = float(os.getenv("CONSTRAINED_JITTER_MS", "30"))
CONSTRAINED_BANDWIDTH = float(os.getenv("CONSTRAINED_BANDWIDTH_KBPS", "512"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(3600.0))
    storage.load_from_json()
    if storage.is_loaded:
        log.info("loaded %d conversations", len(storage.conversations))
    yield
    await app.state.http.aclose()


app = FastAPI(title="benchmark", version="2.0", lifespan=lifespan)
otel_init.instrument_fastapi(app)

storage = LocomoService(
    data_url=os.getenv("DATA_URL", "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json")
)


# ---------------------------- request models ----------------------------

class ExperimentRequest(BaseModel):
    """Standard experiment call: one CONV × one mode × N backends.

    `mode` selects the network regime:
      - "unconstrained": toxiproxy cleared (lat=0, jit=0, bw=0)
      - "constrained":   toxiproxy set to CONSTRAINED_* (defaults: 150/30/512)

    For each backend we wipe memory state, apply toxics, load the conv
    (one /memorize call per message), run the Q&A loop ONCE per question,
    then wipe again. Each answer is graded by `llm_as_judge_seed`
    independent judge calls and majority-voted (>50% CORRECT ⇒ CORRECT,
    default 3 calls so 2/3 = CORRECT).
    """
    conv: int
    mode: str = "unconstrained"
    backends: list[str] = Field(default_factory=lambda: list(VALID_BACKENDS))
    # Number of independent LLM-as-judge calls per (question, answer)
    # whose verdicts are majority-voted into the final `judge` column.
    # Was previously called `seeds` and looped the entire ask phase;
    # now it loops only the judge.
    llm_as_judge_seed: int = 3
    limit: int | None = None
    load_limit: int | None = None
    name_prefix: str = ""
    include_adversarial: bool = False
    # Question filters (applied in order, all combinable):
    #   - question_types: only ask questions whose category is in this list
    #     (LoCoMo categories 1=single-hop, 2=temporal, 3=multi-hop,
    #     4=open-domain, 5=adversarial). When None, all categories pass.
    #   - q_per_type:     keep only the first N questions per category in
    #     qa-list order. When None, no per-category cap is applied.
    # Together these reproduce the calibration setup (e.g. "first 3
    # questions in cats 1-4" = question_types=[1,2,3,4], q_per_type=3).
    # The operator pairs them with `load_limit` to bound the load, and is
    # responsible for ensuring the first `load_limit` messages contain
    # the evidence for every selected question.
    question_types: list[int] | None = None
    q_per_type: int | None = None
    # When True, skip the post-leg memory wipe so the next call with the
    # same name_prefix can resume against an already-loaded backend. The
    # pre-leg reset is already resume-aware, so the load is paid once
    # and reused thereafter — useful for fast iteration on retrieval.
    skip_post_reset: bool = False
    # Explicit session_id to resume INTO. When provided, the bench keys
    # resume on this session_id instead of auto-generating a fresh one,
    # so a re-run with the same id continues the existing run (skipping
    # rows already present for that session_id) and a re-run without it
    # — or with a different id — starts clean. When None (default), a
    # new run_id is generated per /experiment call: every run is its
    # own session, prior CSV rows from other sessions are not skipped.
    session_id: str | None = None

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("unconstrained", "constrained"):
            raise ValueError("mode must be 'unconstrained' or 'constrained'")
        return v

    @field_validator("backends")
    @classmethod
    def _check_backends(cls, v: list[str]) -> list[str]:
        bad = [b for b in v if b not in VALID_BACKENDS]
        if bad:
            raise ValueError(f"unknown backend(s) {bad}; expected subset of {list(VALID_BACKENDS)}")
        if not v:
            raise ValueError("backends must be non-empty")
        return v


# ---------------------------- helpers -----------------------------------

def _toxics_for(mode: str) -> tuple[float, float, float]:
    # `mode` is constrained by ExperimentRequest's validator, so the
    # branches are exhaustive — no silent (0,0,0) fallback for unexpected
    # values, which would otherwise mask validator regressions.
    if mode == "constrained":
        return CONSTRAINED_LATENCY, CONSTRAINED_JITTER, CONSTRAINED_BANDWIDTH
    if mode == "unconstrained":
        return 0.0, 0.0, 0.0
    raise ValueError(f"unexpected mode {mode!r}")


def _session_date_for_question(conv, evidence: list | None) -> str:
    """Resolve evidence like ['D1:3'] to the session's date string. The
    responder uses this as its conversation-date anchor so it emits
    absolute dates instead of relative references ("yesterday", "last
    week"); the judge no longer needs an anchor since responder output
    should already be in absolute form."""
    if not evidence or conv is None:
        return ""
    sds = getattr(conv, "session_datetimes", None) or {}
    for ev in evidence:
        if not isinstance(ev, str) or not ev.startswith("D"):
            continue
        num = ev[1:].split(":", 1)[0]
        if num.isdigit():
            d = sds.get(f"session_{num}_date_time")
            if d:
                return d
    return ""


def _config_slug(backend: str, conv: int, mode: str) -> str:
    return f"{backend}_conv{conv}_{mode}"


def _experiment_name(backend: str, conv: int, mode: str, prefix: str = "") -> str:
    """Single experiment-name builder — the same string is used for
    warmup, load, and ask rows; phase lives in the `phase` column."""
    return f"{prefix}{_config_slug(backend, conv, mode)}"


# ---------------------------- routes ------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "loaded_conversations": len(storage.conversations),
        "results_dir": str(RESULTS_DIR),
        "toxiproxy_admins": list(TOXIPROXY_ADMINS),
        "constrained_profile": {
            "latency_ms": CONSTRAINED_LATENCY,
            "jitter_ms": CONSTRAINED_JITTER,
            "bandwidth_kbps": CONSTRAINED_BANDWIDTH,
        },
    }


async def _do_warmup(
    client: httpx.AsyncClient,
    backend: str,
    conv_idx: int,
    mode: str,
    latency: float,
    jitter: float,
    bandwidth: float,
    session_id: str,
    name_prefix: str = "",
):
    """Pay backend one-time init (graphiti indices, qdrant collections)
    after the pre-leg reset. Emits a single phase=warmup row so the cost
    is visible but separated from per-message work. Reset before this
    guarantees the cost is honestly re-incurred per leg."""
    t_call_start = time.monotonic()
    snap_t0 = await cgroup_snapshot(client)
    err: str | None = None
    detail: dict[str, Any] = {}
    with _tracer.start_as_current_span(
        "warmup",
        context=OTelContext(),
        attributes={
            "langfuse.session.id": session_id,
            "langfuse.tags": f"service:benchmark,memory:{backend},conv:{conv_idx},mode:{mode}",
            "dmas.backend": backend, "dmas.conv_index": conv_idx,
        },
    ) as warmup_span:
        trace_id = format_trace_id(warmup_span.get_span_context().trace_id)
        try:
            detail = await post_warmup(client, backend, conv_idx)
        except Exception as exc:
            err = str(exc)[:300]
            log.exception("[warmup] post failed")

    t_call_end = time.monotonic()
    snap_t1 = await cgroup_wait_io_quiet(client)
    t_after_flush = time.monotonic()
    compute_ms = (t_call_end - t_call_start) * 1000.0
    flush_ms = (t_after_flush - t_call_end) * 1000.0
    wall_ms = compute_ms + flush_ms
    mdelta = cgroup_delta(snap_t0, snap_t1)

    row: dict[str, Any] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_name": _experiment_name(backend, conv_idx, mode, name_prefix),
        "phase": "warmup",
        "memory": backend,
        "mode": mode,
        "conversation_index": conv_idx,
        # `name_prefix` is consumed by results._file_for_row for filename
        # routing; DictWriter drops it because it isn't in COLUMNS.
        "name_prefix": name_prefix,
        "seed": None,
        "question": None,
        "answer": (json.dumps(detail) if detail else None),
        "gold_answer": None,
        "question_category": None,
        "judge_verdict": None,
        "judge_reason": None,
        "toxic_latency": latency, "toxic_jitter": jitter, "toxic_bandwidth": bandwidth,
        "wall_ms": wall_ms,
        "compute_ms": compute_ms,
        "flush_ms": flush_ms,
        **mdelta,
        "edge_llm_tokens": 0, "edge_llm_cost_usd": 0.0,
        "cloud_llm_tokens": 0, "cloud_llm_cost_usd": 0.0,
        "llm_tokens": 0, "llm_cost_usd": 0.0,
        "error": err,
    }
    append_row(row)
    log.info("[warmup] %s wall=%.2fs (compute=%.2fs flush=%.2fs)%s",
             backend, wall_ms / 1000, compute_ms / 1000, flush_ms / 1000,
             f" ERROR={err}" if err else "")
    yield {"_phase": "warmup_done", "backend": backend, "conv": conv_idx, "mode": mode,
           "wall_ms": wall_ms, "compute_ms": compute_ms, "flush_ms": flush_ms,
           "status": "error" if err else "ok", **detail}


async def _do_load(
    client: httpx.AsyncClient,
    backend: str,
    conv_idx: int,
    mode: str,
    latency: float,
    jitter: float,
    bandwidth: float,
    skip_messages: set[str],
    session_id: str,
    load_limit: int | None = None,
    name_prefix: str = "",
):
    """Load CONV into BACKEND one message at a time.

    The bench splits each session's turns into individual messages, POSTs
    each to coordinator/memorize, and writes one `phase=load` row per
    confirmed save. `question` carries `<session>:<msg_idx>` — that pair
    is the resume marker.
    """
    conv = storage.get_conversation_by_index(conv_idx)
    if conv is None:
        raise HTTPException(404, f"conv {conv_idx} not found")

    keys = sorted(
        (k for k in conv.sessions if k.startswith("session_") and k[len("session_"):].isdigit()),
        key=lambda k: int(k[len("session_"):]),
    )

    # Flatten conv → ordered list of (session_key, msg_idx, turn) tuples.
    msgs: list[tuple[str, int, Any]] = []
    for sk in keys:
        for mi, turn in enumerate(conv.sessions[sk]):
            msgs.append((sk, mi, turn))

    if load_limit is not None and load_limit > 0:
        msgs = msgs[:load_limit]

    log.info("[load] backend=%s conv=%d mode=%s sessions=%d messages=%d already_done=%d",
             backend, conv_idx, mode, len(keys), len(msgs), len(skip_messages))

    totals = {
        "added": 0, "failed": 0, "wall_ms": 0.0,
        "edge_llm_tokens": 0, "edge_llm_cost_usd": 0.0,
        "cloud_llm_tokens": 0, "cloud_llm_cost_usd": 0.0,
        "llm_tokens": 0, "llm_cost_usd": 0.0,
        "messages_loaded": 0, "messages_skipped": 0,
    }

    # `langfuse.session.id` is set on every load.message + ask.question
    # span and groups all traces from this run's leg into one Langfuse
    # session. session_id is `{experiment_name}_{run_id}` — the run_id
    # suffix is generated per /experiment call so re-runs of the same
    # config land in distinct sessions instead of stacking up in one.
    # Single parent `load` span so the operator sees one collapsible
    # entry in the session view instead of N per-message session-roots.
    # Each `load.message` below nests under it via the active context.
    # `load_summaries` accumulates one entry per persisted message so
    # the parent span can surface the full list as its `output` —
    # mirroring how `responder.respond` exposes every search call. Lets
    # the operator open the load span in Langfuse and inspect every
    # message that was saved, instead of seeing an empty I/O panel.
    load_summaries: list[dict[str, Any]] = []
    load_input = {
        "backend": backend,
        "conv_index": conv_idx,
        "mode": mode,
        "speakers": [conv.speaker_a, conv.speaker_b],
        "sessions": len(keys),
        "messages_total": len(msgs),
        "messages_already_loaded": len(skip_messages),
    }
    with _tracer.start_as_current_span(
        "load",
        context=OTelContext(),
        attributes={
            "langfuse.session.id": session_id,
            "langfuse.tags": f"service:benchmark,memory:{backend},conv:{conv_idx},mode:{mode}",
            "dmas.backend": backend, "dmas.conv_index": conv_idx,
            "dmas.total_msgs": len(msgs),
            "input": json.dumps(load_input)[:8000],
        },
    ) as load_sp:
      for ord_idx, (sk, mi, turn) in enumerate(msgs, start=1):
        marker = str(ord_idx)
        session_int = int(sk[len("session_"):])
        # Visible progress for tmux: "[load] mem0 23/419 ..."
        if marker in skip_messages:
            totals["messages_skipped"] += 1
            log.info("[load] %s %d/%d session=%d SKIP (in per-experiment CSV)",
                     backend, ord_idx, len(msgs), session_int)
            yield {"_phase": "memory_skipped", "backend": backend, "conv": conv_idx,
                   "mode": mode, "i": ord_idx, "total": len(msgs),
                   "session": session_int, "msg_idx": mi}
            continue

        # `load.message` nests under the parent `load` span (active in
        # the current context), which keeps every per-message save under
        # one collapsible session entry. Auto-instrumented httpx then
        # opens its `POST /coordinator/memorize` client span as our
        # child, which in turn nests the coordinator's manual
        # `coordinator->memory.memorize` span and the memory-side
        # `memory.persist` + gen_ai children.
        with _tracer.start_as_current_span(
            "load.message",
            attributes={
                "langfuse.session.id": session_id,
                "langfuse.tags": f"service:benchmark,memory:{backend},conv:{conv_idx},mode:{mode}",
                "dmas.backend": backend, "dmas.conv_index": conv_idx,
                "dmas.session": session_int, "dmas.msg_idx": mi,
                "dmas.ord_idx": ord_idx, "dmas.total_msgs": len(msgs),
            },
        ) as msg_span:
            msg_trace_id = format_trace_id(msg_span.get_span_context().trace_id)
            dt_key = f"{sk}_date_time"
            session_dt = conv.session_datetimes.get(dt_key, "")
            data = {
                "speaker_a": conv.speaker_a,
                "speaker_b": conv.speaker_b,
                "sessions": {sk: [turn.dict()]},
                "session_datetimes": ({dt_key: session_dt} if session_dt else {}),
            }
            msg_span.set_attribute("input", json.dumps(data)[:8000])

            # cgroup counters are read straight from /sys/fs/cgroup pseudo-
            # files — kernel-maintained, every read is the value at the
            # moment of access. No telegraf scrape interval, no docker daemon
            # stats cache, so a snapshot before+after a sub-second call
            # returns the actual delta with no rounding.
            t_call_start = time.monotonic()
            snap_t0 = await cgroup_snapshot(client)
            err: str | None = None
            confirmation: dict[str, Any] = {}
            try:
                confirmation = await post_memorize(client, backend, conv_idx, data,
                                                   session_id=session_id, mode=mode)
            except Exception as exc:
                err = str(exc)[:300]
                log.exception("[load] post failed")

            t_call_end = time.monotonic()
            # Block until cloud-side disk activity quiesces so async DB
            # checkpoints (Neo4j) land in this row's t1 snapshot. Track the
            # extra wait separately as flush_ms so wall_ms can be split into
            # the production-equivalent compute_ms and the artificial flush.
            snap_t1 = await cgroup_wait_io_quiet(client)
            t_after_flush = time.monotonic()
            compute_ms = (t_call_end - t_call_start) * 1000.0
            flush_ms = (t_after_flush - t_call_end) * 1000.0
            wall_ms = compute_ms + flush_ms
            mdelta = cgroup_delta(snap_t0, snap_t1)

            # The memory service returns a `memories` list — for a single-turn
            # payload that's [single_event] (or [] on validation error / 0 on
            # blank turn). Take the first event if any; otherwise synthesise.
            memories_in_resp = list(confirmation.get("memories") or [])
            m_evt = memories_in_resp[0] if memories_in_resp else {
                "session": sk, "chunk_idx": mi,
                "status": "failed" if err else "skipped",
                "preview": "", "error": err,
                "edge_tokens": 0, "edge_cost": 0.0,
                "cloud_tokens": 0, "cloud_cost": 0.0,
            }
            status = m_evt.get("status") or ("failed" if err else "ok")
            # Per-message LLM tokens/cost split by deployment, sourced from
            # litellm's /metrics counter on the memory side. edge_llm_* is
            # ollama-served (free), cloud_llm_* is OpenAI passthrough.
            edge_llm_tokens = int(m_evt.get("edge_tokens") or 0)
            edge_llm_cost = float(m_evt.get("edge_cost") or 0.0)
            cloud_llm_tokens = int(m_evt.get("cloud_tokens") or 0)
            cloud_llm_cost = float(m_evt.get("cloud_cost") or 0.0)
            llm_tokens = edge_llm_tokens + cloud_llm_tokens
            llm_cost = edge_llm_cost + cloud_llm_cost

            row: dict[str, Any] = {
                "session_id": session_id,
                "trace_id": msg_trace_id,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment_name": _experiment_name(backend, conv_idx, mode, name_prefix),
                "phase": "load",
                "memory": backend,
                "mode": mode,
                "conversation_index": conv_idx,
                "name_prefix": name_prefix,
                # seed carries the session number for load rows — keeps
                # `question_category` reserved for the LoCoMo question type
                # used by ask rows so the notebook's category→question_type
                # mapping never sees session ordinals.
                "seed": session_int,
                "question": marker,
                "answer": (m_evt.get("preview") or None),
                "gold_answer": None,
                "question_category": None,
                "judge_verdict": None,
                "judge_reason": None,
                "toxic_latency": latency, "toxic_jitter": jitter, "toxic_bandwidth": bandwidth,
                "wall_ms": wall_ms,
                "compute_ms": compute_ms,
                "flush_ms": flush_ms,
                **mdelta,
                "edge_llm_tokens": edge_llm_tokens,
                "edge_llm_cost_usd": edge_llm_cost,
                "cloud_llm_tokens": cloud_llm_tokens,
                "cloud_llm_cost_usd": cloud_llm_cost,
                "llm_tokens": llm_tokens,
                "llm_cost_usd": llm_cost,
                "error": err or m_evt.get("error"),
            }
            append_row(row)

            msg_span.set_attribute("dmas.status", status)
            msg_span.set_attribute("dmas.wall_ms", wall_ms)
            msg_span.set_attribute("dmas.compute_ms", compute_ms)
            msg_span.set_attribute("dmas.flush_ms", flush_ms)
            msg_span.set_attribute("dmas.llm_tokens", llm_tokens)
            msg_span.set_attribute("dmas.llm_cost_usd", llm_cost)
            msg_span.set_attribute(
                "output",
                json.dumps({
                    "status": status, "wall_ms": wall_ms,
                    "preview": (m_evt.get("preview") or "")[:500],
                    "llm_tokens": llm_tokens, "llm_cost_usd": llm_cost,
                })[:8000],
            )
            # Append a per-message entry to the parent `load` span's
            # output payload so the session view exposes the full list
            # of saved messages on the parent span (collapsible JSON in
            # Langfuse), same pattern as `responder.respond`.
            load_summaries.append({
                "ord": ord_idx,
                "session": session_int,
                "msg_idx": mi,
                "status": status,
                "wall_ms": wall_ms,
                "preview": (m_evt.get("preview") or "")[:300],
                "llm_tokens": llm_tokens,
            })

            log.info("[load] %s %d/%d session=%d %s wall=%.2fs edge=%d/$%.4f cloud=%d/$%.4f",
                     backend, ord_idx, len(msgs), session_int,
                     status.upper(), wall_ms / 1000,
                     edge_llm_tokens, edge_llm_cost, cloud_llm_tokens, cloud_llm_cost)
            yield {"_phase": "memory_added", "backend": backend, "conv": conv_idx,
                   "mode": mode, "session": session_int, "msg_idx": mi,
                   "msg_counter": ord_idx, "total": len(msgs),
                   "status": status,
                   "wall_ms": wall_ms,
                   "edge_llm_tokens": edge_llm_tokens, "edge_llm_cost_usd": edge_llm_cost,
                   "cloud_llm_tokens": cloud_llm_tokens, "cloud_llm_cost_usd": cloud_llm_cost,
                   "llm_tokens": llm_tokens, "llm_cost_usd": llm_cost}

            if err:
                raise HTTPException(502, f"memorize {marker} failed: {err}")

            if status == "ok":
                totals["added"] += 1
                totals["messages_loaded"] += 1
            elif status == "skipped":
                totals["messages_skipped"] += 1
            else:
                totals["failed"] += 1
            totals["wall_ms"] += wall_ms
            totals["edge_llm_tokens"] += edge_llm_tokens
            totals["edge_llm_cost_usd"] += edge_llm_cost
            totals["cloud_llm_tokens"] += cloud_llm_tokens
            totals["cloud_llm_cost_usd"] += cloud_llm_cost
            totals["llm_tokens"] += llm_tokens
            totals["llm_cost_usd"] += llm_cost
      try:
          load_sp.set_attribute("dmas.messages_loaded", totals["messages_loaded"])
          load_sp.set_attribute("dmas.messages_skipped", totals["messages_skipped"])
          load_sp.set_attribute("dmas.messages_failed", totals["failed"])
          load_sp.set_attribute("dmas.wall_ms", totals["wall_ms"])
          load_sp.set_attribute(
              "output",
              json.dumps({
                  "messages_total": len(msgs),
                  "messages_loaded": totals["messages_loaded"],
                  "messages_skipped": totals["messages_skipped"],
                  "messages_failed": totals["failed"],
                  "wall_ms": totals["wall_ms"],
                  "llm_tokens": totals["llm_tokens"],
                  "llm_cost_usd": totals["llm_cost_usd"],
                  "messages": load_summaries,
              })[:64000],
          )
      except Exception:
          pass

    yield {"_phase": "load_done", "backend": backend, "conv": conv_idx, "mode": mode,
           "messages_total": len(msgs), "session_id": session_id, **totals}


@app.post("/experiment")
async def experiment(req: ExperimentRequest, request: Request):
    if not storage.is_loaded:
        raise HTTPException(503, "dataset not loaded")
    if storage.get_conversation_by_index(req.conv) is None:
        raise HTTPException(404, f"conv {req.conv} not found")

    client: httpx.AsyncClient = request.app.state.http
    latency, jitter, bandwidth = _toxics_for(req.mode)

    questions = storage.get_conversation_questions_by_index(req.conv)
    if req.question_types is not None or req.q_per_type is not None:
        types_filter = set(req.question_types) if req.question_types else None
        per_type_cap = req.q_per_type
        seen: dict[int, int] = {}
        filtered: list[dict[str, Any]] = []
        for q in questions:
            cat = q.get("category")
            if types_filter is not None and cat not in types_filter:
                continue
            if per_type_cap is not None and seen.get(cat, 0) >= per_type_cap:
                continue
            filtered.append(q)
            seen[cat] = seen.get(cat, 0) + 1
        questions = filtered
    if not req.include_adversarial:
        questions = [q for q in questions if q.get("category") != 5]
    if req.limit is not None and req.limit > 0:
        questions = questions[: req.limit]

    # One run_id per /experiment call. Suffixed onto every leg's
    # session_id so re-running the same (backend, conv, mode, prefix)
    # gets a fresh Langfuse session instead of stacking traces under
    # the previous run's session. 8 hex chars ≈ 4 billion combinations,
    # plenty for human-distinguishable URLs.
    # If the operator passes `session_id` on the request, we use it
    # verbatim and derive a stable run_id from its suffix so later legs
    # in the same call still group together.
    run_id = (req.session_id.rsplit("_", 1)[-1]
              if req.session_id else uuid.uuid4().hex[:8])
    log.info(
        "/experiment conv=%d mode=%s backends=%s judge_n=%d questions=%d toxics=lat%s/jit%s/bw%s run_id=%s",
        req.conv, req.mode, req.backends, req.llm_as_judge_seed, len(questions), latency, jitter, bandwidth, run_id,
    )

    async def gen():
        for backend in req.backends:
            exp_name = _experiment_name(backend, req.conv, req.mode, req.name_prefix)
            # Use the operator-supplied session_id verbatim only when it
            # matches THIS leg's exp_name; otherwise rebuild from
            # exp_name + run_id so each (backend, conv, mode) leg in a
            # multi-leg call still gets its own distinct session.
            if req.session_id and req.session_id.startswith(exp_name + "_"):
                session_id = req.session_id
            else:
                session_id = f"{exp_name}_{run_id}"
            yield json.dumps({"_phase": "backend_start", "backend": backend,
                              "mode": req.mode, "conv": req.conv,
                              "session_id": session_id}) + "\n"

            # Apply toxics first so the proxies exist before reset routes through them.
            try:
                await apply_all(client, TOXIPROXY_ADMINS, latency, jitter, bandwidth)
            except HTTPException as exc:
                yield json.dumps({"_phase": "toxics_error", "backend": backend,
                                  "status": exc.status_code, "detail": str(exc.detail)[:300]}) + "\n"
                continue

            # Resume-aware reset: only wipe state when no messages have
            # been saved yet for this (backend, conv, mode). On a partial
            # resume, the storage on disk is the source of truth — wiping
            # would discard the work the existing CSV rows attest to.
            done_msgs = loaded_messages(backend, req.conv, latency, jitter, bandwidth,
                                        req.name_prefix, session_id=session_id)
            if not done_msgs:
                try:
                    pre = await post_reset(client, backend)
                    yield json.dumps({"_phase": "reset_pre", "backend": backend, **pre}) + "\n"
                except Exception as exc:
                    yield json.dumps({"_phase": "reset_error", "when": "pre",
                                      "backend": backend, "error": str(exc)[:300]}) + "\n"
                    continue

                # Warmup — pay one-time backend init (graphiti index build,
                # qdrant collection create) up front so it gets its own row
                # instead of inflating row #1 of the load. Reset above
                # guarantees a clean slate so the cost is honest per leg.
                try:
                    async for evt in _do_warmup(client, backend, req.conv, req.mode,
                                                latency, jitter, bandwidth,
                                                session_id,
                                                name_prefix=req.name_prefix):
                        yield json.dumps(evt) + "\n"
                except Exception as exc:
                    yield json.dumps({"_phase": "warmup_error", "backend": backend,
                                      "error": str(exc)[:300]}) + "\n"
            else:
                yield json.dumps({"_phase": "reset_pre_skipped", "backend": backend,
                                  "reason": "resuming",
                                  "already_loaded": len(done_msgs)}) + "\n"

            # Load — one message per call, one row per message in the per-experiment CSV.
            try:
                async for evt in _do_load(client, backend, req.conv, req.mode,
                                          latency, jitter, bandwidth, done_msgs,
                                          session_id,
                                          load_limit=req.load_limit,
                                          name_prefix=req.name_prefix):
                    yield json.dumps(evt) + "\n"
            except HTTPException as exc:
                yield json.dumps({"_phase": "load_error", "backend": backend,
                                  "status": exc.status_code, "detail": str(exc.detail)[:300]}) + "\n"
                continue

            # Ask. Each question is asked ONCE; only the LLM-as-judge
            # repeats `req.llm_as_judge_seed` times and majority-votes.
            skip = already_done(backend, req.conv, latency, jitter, bandwidth,
                                req.name_prefix, session_id=session_id)
            conv_obj = storage.get_conversation_by_index(req.conv)
            n_emitted = n_skipped = 0
            for i, q in enumerate(questions, start=1):
                qtext = q["question"]
                if qtext in skip:
                    n_skipped += 1
                    log.info("[exp] %s %d/%d SKIP | %s",
                             backend, i, len(questions), qtext[:120].replace("\n", " "))
                    continue
                # Re-verify toxics so external drift mid-run aborts cleanly.
                await verify_all(client, TOXIPROXY_ADMINS, latency, jitter, bandwidth)
                log.info("[exp] %s %d/%d ASK | %s",
                         backend, i, len(questions), qtext[:200].replace("\n", " "))
                # Resolve the conversation date for this question's evidence
                # BEFORE asking so we can pass it to the responder as the
                # anchor for relative time references ("yesterday", "last
                # week").
                session_date = _session_date_for_question(conv_obj, q.get("evidence"))
                t_call_start = time.monotonic()
                snap_t0 = await cgroup_snapshot(client)
                usage_t0 = await litellm_usage_snapshot(client)

                answer = ""
                err: str | None = None
                memories_returned: int | None = None
                top_k_seen: int | None = None
                search_calls: int | None = None
                responder_context_window_tokens: int | None = None
                responder_context_window_cost_usd: float | None = None
                # Detached root span: each ask.question gets its own
                # trace_id; langfuse.session.id ties it to the same
                # session as the load.message traces above.
                with _tracer.start_as_current_span(
                    "ask.question",
                    context=OTelContext(),
                    attributes={
                        "langfuse.session.id": session_id,
                        "langfuse.tags": f"service:benchmark,memory:{backend},conv:{req.conv},mode:{req.mode}",
                        "input": qtext[:8000],
                        "dmas.backend": backend, "dmas.conv_index": req.conv,
                        "dmas.mode": req.mode, "dmas.question_idx": i,
                    },
                ) as ask_span:
                    ask_trace_id = format_trace_id(ask_span.get_span_context().trace_id)
                    try:
                        resp = await post_ask(client, qtext, backend, session_date,
                                              session_id=session_id,
                                              conv_index=req.conv, mode=req.mode)
                        # Strip trailing whitespace — the responder/SLM
                        # sometimes emits a trailing newline that breaks
                        # CSV viewers even though it's properly quoted.
                        answer = (resp.get("answer", "") or "").strip()
                        if resp.get("status") == "error":
                            err = resp.get("error")
                        # Retrieval stats threaded up from
                        # coordinator → responder → memory. Optional;
                        # legacy responses that don't carry them leave
                        # the columns blank.
                        if "memories_returned" in resp:
                            memories_returned = int(resp["memories_returned"] or 0)
                        if "top_k" in resp and resp["top_k"] is not None:
                            top_k_seen = int(resp["top_k"])
                        if "search_calls" in resp:
                            search_calls = int(resp["search_calls"] or 0)
                        if resp.get("responder_context_window_tokens") is not None:
                            responder_context_window_tokens = int(resp["responder_context_window_tokens"])
                        if resp.get("responder_context_window_cost_usd") is not None:
                            responder_context_window_cost_usd = float(resp["responder_context_window_cost_usd"])
                    except Exception as exc:
                        err = str(exc)[:300]
                        # Mark the span as errored so Langfuse renders it
                        # with the failure icon. record_exception attaches
                        # the traceback as a span event for full context.
                        ask_span.record_exception(exc)
                        ask_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, err))
                    try:
                        ask_span.set_attribute("output", (answer or "")[:8000])
                        if err:
                            ask_span.set_attribute("dmas.error", err)
                    except Exception:
                        pass

                t_call_end = time.monotonic()
                snap_t1 = await cgroup_wait_io_quiet(client)
                t_after_flush = time.monotonic()
                compute_ms = (t_call_end - t_call_start) * 1000.0
                flush_ms = (t_after_flush - t_call_end) * 1000.0
                wall_ms = compute_ms + flush_ms
                usage_t1 = await litellm_usage_snapshot(client)
                mdelta = cgroup_delta(snap_t0, snap_t1)
                du = litellm_usage_diff(usage_t0, usage_t1)
                edge_llm_tokens = du["edge_tokens"]
                edge_llm_cost = du["edge_cost"]
                cloud_llm_tokens = du["cloud_tokens"]
                cloud_llm_cost = du["cloud_cost"]
                llm_tokens = edge_llm_tokens + cloud_llm_tokens
                llm_cost = edge_llm_cost + cloud_llm_cost

                gold = q.get("answer")
                judge_trace_id: str | None = None
                if err:
                    j_label = None
                    j_reason = None
                    j_n = 0
                    j_correct = 0
                else:
                    # Detached root: each judge.evaluate is its own trace
                    # under the same session as ask.question + the load
                    # legs. `contextvars.copy_context().run` propagates
                    # the OTel current span into the worker thread so
                    # the N parallel `gpt-4o-mini` judge calls nest
                    # under judge.evaluate instead of forming orphan
                    # gen_ai traces.
                    with _tracer.start_as_current_span(
                        "judge.evaluate",
                        context=OTelContext(),
                        attributes={
                            "langfuse.session.id": session_id,
                            "langfuse.tags": f"service:judge,memory:{backend},conv:{req.conv},mode:{req.mode}",
                            "input": json.dumps({
                                "question": qtext, "gold": gold, "answer": answer,
                            })[:8000],
                            "dmas.judge_n": req.llm_as_judge_seed,
                        },
                    ) as judge_span:
                        judge_trace_id = format_trace_id(judge_span.get_span_context().trace_id)
                        ctx = contextvars.copy_context()
                        j = await asyncio.to_thread(
                            ctx.run, judge_majority,
                            qtext, gold, answer, req.llm_as_judge_seed,
                        )
                        try:
                            judge_span.set_attribute("output", json.dumps({
                                "verdict": j.label,
                                "correct_votes": j.correct_votes,
                                "n": j.n,
                                "reason": (j.reasonings[0] if j.reasonings else "")[:2000],
                            })[:8000])
                        except Exception:
                            pass
                    j_label = j.label
                    j_reason = (j.reasonings[0] if j.reasonings else "").strip()
                    j_n = j.n
                    j_correct = j.correct_votes

                row: dict[str, Any] = {
                    "session_id": session_id,
                    "trace_id": ask_trace_id,
                    "judge_trace_id": judge_trace_id,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "experiment_name": exp_name,
                    "phase": "ask",
                    "memory": backend,
                    "mode": req.mode,
                    "conversation_index": req.conv,
                    "name_prefix": req.name_prefix,
                    # `seed` only carries meaning on load rows (session
                    # number). Ask rows now run once per question, so
                    # leave the column null here.
                    "seed": None,
                    "question": qtext,
                    "answer": answer,
                    "gold_answer": gold,
                    "question_category": q.get("category"),
                    "judge_n": j_n,
                    "judge_verdict": j_label,
                    "judge_reason": j_reason,
                    "toxic_latency": latency,
                    "toxic_jitter": jitter,
                    "toxic_bandwidth": bandwidth,
                    "wall_ms": wall_ms,
                    "compute_ms": compute_ms,
                    "flush_ms": flush_ms,
                    **mdelta,
                    "edge_llm_tokens": edge_llm_tokens,
                    "edge_llm_cost_usd": edge_llm_cost,
                    "cloud_llm_tokens": cloud_llm_tokens,
                    "cloud_llm_cost_usd": cloud_llm_cost,
                    "llm_tokens": llm_tokens,
                    "llm_cost_usd": llm_cost,
                    "memories_returned": memories_returned,
                    "top_k": top_k_seen,
                    "search_calls": search_calls,
                    "responder_context_window_tokens": responder_context_window_tokens,
                    "responder_context_window_cost_usd": responder_context_window_cost_usd,
                    "error": err,
                }
                append_row(row)
                n_emitted += 1
                log.info(
                    "[exp] %s %d/%d DONE judge=%s (%d/%d) wall=%.2fs | gold=%s | answer=%s",
                    backend, i, len(questions), j_label or "ERROR",
                    j_correct, j_n, wall_ms / 1000,
                    str(gold)[:160].replace("\n", " "),
                    answer[:160].replace("\n", " "),
                )
                yield json.dumps(row, default=str) + "\n"

            # Wipe after so the next backend / next call sees clean state,
            # unless the caller asked to keep it (calibration reuses the
            # load across iterations).
            if req.skip_post_reset:
                yield json.dumps({"_phase": "reset_post_skipped", "backend": backend,
                                  "reason": "skip_post_reset"}) + "\n"
            else:
                try:
                    post = await post_reset(client, backend)
                    yield json.dumps({"_phase": "reset_post", "backend": backend, **post}) + "\n"
                except Exception as exc:
                    yield json.dumps({"_phase": "reset_error", "when": "post",
                                      "backend": backend, "error": str(exc)[:300]}) + "\n"

            yield json.dumps({"_phase": "backend_done", "backend": backend,
                              "emitted": n_emitted, "skipped": n_skipped}) + "\n"

        yield json.dumps({"_end": True}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
