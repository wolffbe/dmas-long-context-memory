"""Coordinator HTTP API: /load and /experiment.

Both endpoints take a self-describing JSON body, do their full job server-side,
and return a single JSON response (no streaming). Per-question progress is
written to coordinator stdout — visible via `make logs`.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import string
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agents import post_ask, post_load
from app.config import CFG
from app.datasets import FlatTurn, datasets, flatten_locomo, flatten_longmemeval
from app.judges import judge_for_backend, set_judge_model
from app.system_metrics import delta as metrics_delta
from app.system_metrics import peak_ram, snapshot
from app.toxics import sync_all

router = APIRouter()
log = logging.getLogger("coord.routes")


# ---------- LLM normalization (kept identical to old toxiproxy-proxy) ----------

REQUIRED_AGENT_IDS = ("1", "2", "3")
LOCOMO_CATEGORY_LABEL = {1: "single_hop", 2: "temporal", 3: "open_domain", 4: "multi_hop", 5: "adversarial"}


def _category_label(cat: Any) -> str:
    try:
        return LOCOMO_CATEGORY_LABEL.get(int(cat), "unknown")
    except (TypeError, ValueError):
        return "unknown"


# ---------- token-F1 (formerly in experiments/lib/drivers.py) ----------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def _normalize(s: str) -> list[str]:
    s = (s or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def _token_f1(prediction: str, gold: str) -> float:
    pred_toks = _normalize(prediction)
    gold_toks = _normalize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common: dict[str, int] = {}
    for t in pred_toks:
        if t in gold_toks and gold_toks.count(t) > common.get(t, 0):
            common[t] = common.get(t, 0) + 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _string_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower().strip(), (b or "").lower().strip()).ratio()


# ---------- Pydantic ----------

class LoadRequest(BaseModel):
    backend: str
    dataset: str
    conv: int | None = None
    qid: str | None = None
    all: bool = False
    limit: int | None = None  # smoke-test: truncate to first N turns


class ExperimentRequest(BaseModel):
    backend: str
    dataset: str
    conv: int | None = None
    qid: str | None = None
    all: bool = False
    limit: int | None = None  # smoke-test: truncate to first N questions
    latency: float = 0.0
    jitter: float = 0.0
    bandwidth: float = 0.0
    peer_threshold_ms: float = 0.0
    seeds: int = 3
    models: dict[str, str] = Field(default_factory=dict)
    judge_model: str | None = None
    name_prefix: str = ""  # prepended to experiment_name (e.g. "test_" for smoke runs)
    # repro fingerprints supplied by the driver — passed through into each row
    context: dict[str, Any] = Field(default_factory=dict)


# ---------- helpers ----------

def _validate_models(models: dict[str, str]) -> None:
    missing = [a for a in REQUIRED_AGENT_IDS if not models.get(a)]
    if missing:
        raise HTTPException(400, f"models must specify an LLM for every agent; missing: {missing}")


def _validate_locator(req: LoadRequest | ExperimentRequest) -> None:
    if req.dataset not in ("locomo", "longmemeval"):
        raise HTTPException(400, f"dataset must be locomo or longmemeval (got {req.dataset!r})")
    if req.backend not in ("mem0", "zep", "rag", "none"):
        raise HTTPException(400, f"backend must be mem0|zep|rag|none (got {req.backend!r})")
    if req.dataset == "locomo" and req.conv is None and not req.all:
        raise HTTPException(400, "conv or all is required for locomo")
    if req.dataset == "longmemeval" and not (req.qid or req.all):
        raise HTTPException(400, "qid or all is required for longmemeval")


def _collect_turns(req: LoadRequest | ExperimentRequest) -> list[FlatTurn]:
    if req.dataset == "locomo":
        if req.conv is not None:
            return flatten_locomo(datasets.locomo_conversation(req.conv))
        # all conversations, in order
        out: list[FlatTurn] = []
        for i in range(len(datasets.locomo)):
            out.extend(flatten_locomo(datasets.locomo_conversation(i)))
        return out
    if req.qid:
        return flatten_longmemeval(datasets.lme_question(req.qid))
    out = []
    for entry in datasets.lme_index():
        out.extend(flatten_longmemeval(datasets.lme_question(entry["question_id"])))
    return out


def _build_questions(req: ExperimentRequest) -> list[dict[str, Any]]:
    if req.dataset == "locomo":
        conv_indices = (
            [req.conv] if req.conv is not None
            else list(range(len(datasets.locomo)))
        )
        out: list[dict[str, Any]] = []
        for ci in conv_indices:
            for q in datasets.locomo_questions(ci):
                out.append({
                    "lookup_key": ci,
                    "lookup_kind": "conv_idx",
                    "question": q.get("question", ""),
                    "gold_answer": q.get("answer", ""),
                    "category": q.get("category"),
                    "category_label": _category_label(q.get("category")),
                })
        return out
    if req.qid:
        q = datasets.lme_question(req.qid)
        return [{
            "lookup_key": req.qid,
            "lookup_kind": "question_id",
            "question": q.get("question", ""),
            "gold_answer": q.get("answer", ""),
            "category": q.get("question_type"),
            "category_label": q.get("question_type") or "unknown",
        }]
    return [{
        "lookup_key": entry["question_id"],
        "lookup_kind": "question_id",
        "question": entry.get("question", ""),
        "gold_answer": entry.get("answer", ""),
        "category": entry.get("question_type"),
        "category_label": entry.get("question_type") or "unknown",
    } for entry in datasets.lme_index()]


def _build_toxics(latency: float, jitter: float, bandwidth: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if latency > 0 or jitter > 0:
        out.append({
            "name": "latency", "type": "latency", "stream": "downstream",
            "attributes": {"latency": int(latency), "jitter": int(jitter)},
        })
    if bandwidth > 0:
        out.append({
            "name": "bandwidth", "type": "bandwidth", "stream": "downstream",
            "attributes": {"rate": int(bandwidth)},
        })
    return out


def _experiment_name(req: ExperimentRequest, seed: int) -> str:
    prefix = req.name_prefix or ""
    if not (req.latency or req.jitter or req.bandwidth):
        body = f"{req.backend}_{req.dataset}_unconstrained_seed{seed}"
    else:
        body = (
            f"{req.backend}_{req.dataset}_"
            f"lat{int(req.latency)}_jit{int(req.jitter)}_"
            f"bw{int(req.bandwidth)}_seed{seed}"
        )
    return f"{prefix}{body}"


# ---------- routes ----------

@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "locomo_loaded": len(datasets.locomo),
        "longmemeval_loaded": len(datasets.lme),
        "agents": list(CFG.agent_urls),
        "toxiproxy_admins": list(CFG.toxiproxy_admins),
    }


@router.post("/load")
async def load(req: LoadRequest, request: Request) -> dict:
    _validate_locator(req)
    if not CFG.agent_urls:
        raise HTTPException(500, "no agent URLs configured (env AGENT_URLS)")
    client = request.app.state.http
    turns = _collect_turns(req)
    if req.limit is not None and req.limit > 0:
        turns = turns[: req.limit]
    slug = (
        f"conv_{req.conv}" if req.dataset == "locomo"
        else (req.qid or "all")
    )
    log.info(f"/load backend={req.backend} dataset={req.dataset} slug={slug} turns={len(turns)} limit={req.limit}")

    totals: dict[str, int] = defaultdict(int)
    per_agent: list[int] = [0, 0, 0]
    width = len(str(len(turns)))
    t0 = time.monotonic()

    for n, t in enumerate(turns, start=1):
        idx = t.turn_index % 3
        per_agent[idx] += 1
        item = t.to_item()
        ts0 = time.monotonic()
        try:
            r = await post_load(client, CFG.agent_urls[idx], [item], backend=req.backend)
            added = int(r.get("added", 0))
            failed = int(r.get("failed", 0))
            totals["added"] += added
            totals["failed"] += failed
            status = "ok" if failed == 0 else "FAIL"
        except Exception as exc:
            totals["failed"] += 1
            status = f"ERR {str(exc)[:120]}"
        dt_ms = (time.monotonic() - ts0) * 1000.0
        speaker = item.get("speaker") or item.get("role") or "?"
        marker = item.get("dia_id") or item.get("question_id") or f"turn{t.turn_index}"
        text = (item.get("text") or item.get("content") or "").replace("\n", " ").strip()
        snippet = (text[:60] + "…") if len(text) > 60 else text
        log.info(
            f"[load] {n:>{width}}/{len(turns)} -> agent-{idx+1}  "
            f"speaker={speaker} id={marker}  text={snippet!r}  {status} ({dt_ms:.0f}ms)"
        )

    elapsed = time.monotonic() - t0
    log.info(f"[load] done in {elapsed:.1f}s — added={totals['added']} failed={totals['failed']} per_agent={per_agent}")
    return {
        "backend": req.backend,
        "dataset": req.dataset,
        "slug": slug,
        "turns": len(turns),
        "added": totals["added"],
        "failed": totals["failed"],
        "per_agent": per_agent,
        "elapsed_s": round(elapsed, 3),
    }


@router.post("/experiment")
async def experiment(req: ExperimentRequest, request: Request) -> dict:
    _validate_locator(req)
    _validate_models(req.models)
    if req.judge_model:
        set_judge_model(req.judge_model)

    experiment_t0 = time.monotonic()
    client = request.app.state.http
    judge_fn = judge_for_backend(req.backend)
    questions = _build_questions(req)
    if req.limit is not None and req.limit > 0:
        questions = questions[: req.limit]
    if req.dataset == "locomo":
        slug = f"conv_{req.conv}" if req.conv is not None else "all"
    else:
        slug = req.qid or "all"
    toxics = _build_toxics(req.latency, req.jitter, req.bandwidth)

    log.info(
        f"/experiment backend={req.backend} dataset={req.dataset} slug={slug} "
        f"questions={len(questions)} seeds={req.seeds} "
        f"net=lat={req.latency}/jit={req.jitter}/bw={req.bandwidth} "
        f"peer_threshold={req.peer_threshold_ms}ms"
    )

    # Sync toxics once up front; we re-sync per question only if they need to
    # change (today they don't — same toxics for the whole experiment).
    try:
        await sync_all(client, CFG.toxiproxy_admins, toxics)
    except Exception as exc:
        raise HTTPException(502, f"toxic sync failed: {exc}")

    rows: list[dict[str, Any]] = []
    width = len(str(len(questions)))
    for seed in range(req.seeds):
        exp_name = _experiment_name(req, seed)
        for i, q in enumerate(questions, start=1):
            log.info(
                f"[exp] seed={seed} {i:>{width}}/{len(questions)} "
                f"asking: {q['question'][:60]!r}"
            )

            snap_t0 = await snapshot(client)
            t_call = time.monotonic()
            answer = ""
            m: dict[str, Any] = {}
            try:
                kwargs: dict[str, Any] = {q["lookup_kind"]: q["lookup_key"]}
                resp = await post_ask(
                    client, CFG.upstream_agent_url,
                    question=q["question"], backend=req.backend,
                    llm_model=req.models[CFG.upstream_agent_id], models=req.models,
                    peer_latency_threshold_ms=req.peer_threshold_ms,
                    **kwargs,
                )
                answer = resp.get("answer", "")
                m = resp.get("metrics", {}) or {}
            except Exception as exc:
                m = {"error": str(exc)[:300]}
            wall_ms = (time.monotonic() - t_call) * 1000.0
            snap_t1 = await snapshot(client)
            ram_peak = await peak_ram(client, wall_ms / 1000.0)
            mdelta = metrics_delta(snap_t0, snap_t1, ram_peak)

            judge_label: bool | None = None
            judge_reasoning = ""
            if judge_fn is not None and answer:
                # judges.* are sync (use openai sync client) — run in a thread
                # so we don't block the event loop.
                jr = await asyncio.to_thread(
                    judge_fn, req.dataset, q["question"], q["gold_answer"], answer,
                    question_type=q.get("category"),
                )
                judge_label = jr.label
                judge_reasoning = jr.reasoning

            # Build per-agent token / cost columns. The upstream agent's
            # `accounting` covers itself; `peer_accounting` is keyed by the
            # peers' agent_id.
            own_id = str(CFG.upstream_agent_id)
            buckets: dict[str, dict[str, dict[str, float]]] = {
                own_id: m.get("accounting") or {},
                **(m.get("peer_accounting") or {}),
            }
            cost_cols: dict[str, Any] = {}
            tot_a_tok = 0
            tot_m_tok = 0
            tot_a_cost = 0.0
            tot_m_cost = 0.0
            for aid in ("1", "2", "3"):
                a = (buckets.get(aid, {}) or {}).get("agent") or {}
                mem = (buckets.get(aid, {}) or {}).get("memory") or {}
                a_tok = int(a.get("total_tokens") or 0)
                a_cost = float(a.get("cost_usd") or 0.0)
                m_tok = int(mem.get("total_tokens") or 0)
                m_cost = float(mem.get("cost_usd") or 0.0)
                cost_cols[f"agent{aid}_tokens"] = a_tok
                cost_cols[f"agent{aid}_cost_usd"] = a_cost
                cost_cols[f"agent{aid}_memory_tokens"] = m_tok
                cost_cols[f"agent{aid}_memory_cost_usd"] = m_cost
                tot_a_tok += a_tok
                tot_a_cost += a_cost
                tot_m_tok += m_tok
                tot_m_cost += m_cost
            cost_cols["total_agent_tokens"] = tot_a_tok
            cost_cols["total_agent_cost_usd"] = tot_a_cost
            cost_cols["total_memory_tokens"] = tot_m_tok
            cost_cols["total_memory_cost_usd"] = tot_m_cost
            cost_cols["total_tokens"] = tot_a_tok + tot_m_tok
            cost_cols["total_cost_usd"] = tot_a_cost + tot_m_cost

            row = {
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment_name": exp_name,
                "memory": req.backend,
                "dataset": req.dataset,
                "conversation_index": q["lookup_key"],
                "question_id": q["lookup_key"] if q.get("lookup_kind") == "question_id" else None,
                "question_type": q.get("category") if req.dataset == "longmemeval" else None,
                "seed": seed,
                "agent1_model": req.models["1"],
                "agent2_model": req.models["2"],
                "agent3_model": req.models["3"],
                "question": q["question"],
                "answer": answer,
                "gold_answer": q["gold_answer"],
                "category": q.get("category"),
                "category_label": q.get("category_label"),
                "f1": round(_token_f1(answer, q["gold_answer"]), 4),
                "string_sim": round(_string_sim(answer, q["gold_answer"]), 4),
                "judge_label": judge_label,
                "judge_reasoning": judge_reasoning,
                "toxic_latency": req.latency,
                "toxic_jitter": req.jitter,
                "toxic_bandwidth": req.bandwidth,
                "peer_threshold_ms": req.peer_threshold_ms,
                "measured_latency_ms": m.get("measured_jitter_ms"),
                "peer_help_allowed": m.get("peer_help_allowed"),
                "peers_asked": m.get("peers_asked"),
                "peer_memories": m.get("peer_memories"),
                "own_memories": m.get("own_memories"),
                "judge_model": req.judge_model,
                "error": m.get("error"),
                **cost_cols,
                **mdelta,
                **(req.context or {}),
            }
            rows.append(row)
            log.info(
                f"[exp] seed={seed} {i:>{width}}/{len(questions)} done  "
                f"f1={row['f1']:.3f} wall={wall_ms/1000:.2f}s "
                f"tokens=a1:{row['agent1_tokens']}/a2:{row['agent2_tokens']}/a3:{row['agent3_tokens']} "
                f"cost=${row['agent1_cost_usd']+row['agent2_cost_usd']+row['agent3_cost_usd']:.4f}"
            )

    # Reset toxics on the way out — leave the system clean for the next caller.
    try:
        await sync_all(client, CFG.toxiproxy_admins, [])
    except Exception:
        pass

    experiment_duration_s = round(time.monotonic() - experiment_t0, 3)
    for row in rows:
        row["experiment_duration_s"] = experiment_duration_s
    log.info(f"/experiment finished in {experiment_duration_s}s — {len(rows)} row(s)")
    return {"rows": rows, "n": len(rows), "experiment_duration_s": experiment_duration_s}
