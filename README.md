# Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems

A distributed multi-agent system testbed for benchmarking five long-term conversational memory strategies under different network scenarios.

## Overview

This project compares five approaches to persistent memory in LLM-based multi-agent systems:

| Approach          | Backend                                          | Storage         | What's persisted                                            |
| ----------------- | ------------------------------------------------ | --------------- | ----------------------------------------------------------- |
| **mem0**          | [mem0](https://github.com/mem0ai/mem0) v2.0.1    | Qdrant          | LLM-extracted memory strings (semantic similarity retrieval) |
| **Graphiti**      | [graphiti-core](https://github.com/getzep/graphiti) 0.29 | Neo4j   | Bi-temporal knowledge graph (hybrid edge/node search)       |
| **Cognee**        | [cognee](https://github.com/topoteretes/cognee) 0.5.6 | Neo4j + Qdrant | Generic knowledge graph (graph-completion retrieval)        |
| **RAG**           | in-house (`RagService`)                          | Qdrant          | Verbatim turn text + embedding (cosine top-k retrieval)     |
| **Full Context**  | in-house (`FullContextService`)                  | in-process      | None — full conversation JSON to responder                   |

Memory loading and retrieval mirror the upstream evaluation harnesses verbatim — `mem0ai/memory-benchmarks` for mem0, `getzep/zep-papers` for Graphiti — so this testbed reproduces what each project's authors do, then layers a distributed-systems / cost-tracking framework on top. Cognee's authors don't publish a LoCoMo harness; we drive it per-message to keep the bench's individual-ingestion contract identical across all five backends (documented in `dmas/memory/app/services/cognee_service.py`).

The system evaluates memory retrieval on the [LOCOMO benchmark](https://github.com/snap-research/locomo) — a dataset for very long-term conversational memory in LLM agents.

## Research Context

This repository is the official implementation of a study evaluating long-term memory frameworks in Distributed Multi-Agent Systems (DMAS).

[![arXiv](https://img.shields.io/badge/arXiv-2601.07978-b31b1b.svg)](https://arxiv.org/abs/2601.07978)

While DMAS leverage Large Language Models (LLMs) for collaborative intelligence, systematic evaluations of their memory under network constraints are often lacking. This project addresses that gap by comparing the five backends above on the **LOCOMO** long-context benchmark across unconstrained and constrained network regimes.

The two research questions the testbed answers:
1. Which framework provides the best balance between **knowledge retention**, **computational overhead**, and **financial cost**?
2. How do these metrics vary in a **hybrid cloud–edge environment**?

A **Statistical Pareto Efficiency** framework collapses the trade-off to cost-minimization whenever the accuracy gap fails a two-proportion z-test.

### Citation

<details>
<summary><strong>BibTeX</strong></summary>

```bibtex
@misc{wolff2026costaccuracylongtermmemory,
      title={Cost and accuracy of long-term memory in Distributed Multi-Agent Systems based on Large Language Models},
      author={Benedict Wolff and Jacopo Bennati},
      year={2026},
      eprint={2601.07978},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2601.07978},
}
```

</details>

## Architecture

The stack is a single Docker Compose project (`dmas/docker-compose.yml`):

| Service         | Port      | Group | Description                                                                                            |
| --------------- | --------- | ----- | ------------------------------------------------------------------------------------------------------ |
| **benchmark**   | 8002      | cloud | Owns LOCOMO loading, the Q&A loop, the LLM-as-judge panel, metric capture, per-experiment CSV writes.   |
| **coordinator** | 8003      | edge  | Slim per-question handler — Ollama tool-calling → search\_memory → answer.                              |
| **memory**      | (8005 via toxiproxy) | cloud | mem0, Graphiti, RAG, Cognee, FullContext all instantiated; backend chosen per request.    |
| **responder**   | (8006 via toxiproxy) | cloud | Final-answer generator (`gpt-4o-mini`).                                                    |
| **ollama**      | 11435     | edge  | Local SLM inference (`qwen2.5:3b-instruct-q4_K_M`) for the coordinator.                                  |
| **qdrant**      | 6333      | cloud | Vector store for mem0, RAG, and cognee embeddings.                                                       |
| **neo4j**       | 7474/7687 | cloud | Graph store shared by Graphiti and Cognee. Configured with `db.checkpoint.interval.tx=1` so per-call disk attribution is honest. |
| **litellm**     | 4000      | cloud | Single OpenAI-compatible proxy for both OpenAI and Ollama; emits Prometheus metrics.                    |
| **toxiproxy**   | 8474      | cloud | One container, two named proxies (memory / responder). Toxics set externally; benchmark verifies.       |

### Models

The entire system uses two chat/completion models and one embedding model — nothing else:

| Model | Used by | Where it's pinned |
| --- | --- | --- |
| `gpt-4o-mini` (cloud, `temperature=0` everywhere) | responder, judge, mem0 / graphiti / cognee extractors, graphiti cross-encoder reranker | only cloud chat alias in `dmas/litellm/config.yaml`; `LLM_MODEL` env var. |
| `qwen2.5:3b-instruct-q4_K_M` (edge, ollama-served) | coordinator (SLM tool loop) | only edge alias in `dmas/litellm/config.yaml`; `OLLAMA_MODEL` env var. |
| `text-embedding-3-small` (1536-dim) | mem0, graphiti, cognee, rag | embedding alias in `dmas/litellm/config.yaml`; structurally required (chat models can't produce embeddings). |

The litellm proxy has **no** `openai/*` catch-all — any unexpected model name hard-fails at the proxy with `model not found in model_list` rather than silently routing through.

### Observability

Two layers, kept independent so each can answer different questions:

**Resource accounting (CSVs, kernel counters).** The bench reads counters straight from kernel pseudo-files — no Prometheus, no Telegraf, no scrape interval. Every snapshot reflects the moment of the read.
- **`/sys/fs/cgroup`** (mounted ro) — per-container CPU (`cpu.stat`), RAM (`memory.peak`), disk (`io.stat`). Containers are partitioned via the `group=edge|cloud` label; everything unlabelled (langfuse infra) is excluded so observability never inflates the numbers under test.
- **`/proc/<pid>/net/dev`** (host `/proc` mounted ro) — per-namespace network counters; tx-only, toxiproxy excluded so each byte is counted once at its sender.
- **`/var/run/docker.sock`** (mounted ro) — one-shot container label/PID lookup, cached.
- **`litellm:4000/metrics`** — live tokens/cost counters, split by model name into edge (ollama-served) vs cloud (OpenAI passthrough).

**Tracing (Langfuse v3, OTel).** Self-hosted Langfuse `3.172.1` (web + worker + postgres + clickhouse + redis + minio). The four services (`benchmark`, `coordinator`, `memory`, `responder`) all bootstrap a single shared OTel SDK module — `dmas/shared/otel_init.py`, copied into each image at build time so there's one canonical implementation rather than four hand-kept copies. It exports via OTLP/HTTP to `langfuse-web:3000/api/public/otel/v1/traces`. FastAPI inbound + httpx/requests outbound are auto-instrumented, so cross-service hops stitch into one trace via W3C `traceparent`. LiteLLM's `otel` callback (`litellm/config.yaml`) emits `gen_ai.*` spans for every completion (`gen_ai.system=ollama|openai`, `gen_ai.request.model=…`, token usage), nested under whichever manual span is current.

Each `/experiment` leg shows up as one Langfuse session with three top-level (detached-root) traces:
- **`warmup`** — one trace per leg with the cold-start cost.
- **`load`** — one trace covering the whole message loop. Children are the per-message `load.message` spans; the parent's `output` carries a `messages` array (status, wall_ms, preview, llm_tokens per saved message) so the operator sees the entire load as one collapsible JSON dropdown.
- **`ask.question`** — one trace per question, with `responder.respond` as a sibling root holding the responder's full picture: question, final answer, every search call's query+memories (`retrieved_memories`), and the full LLM `response_prompt` that produced the answer. The backend search lives nested as `mem0.search` / `graphiti.search` / `cognee.search` / `rag.search` / `full_context.dump`. Persistence on the load side similarly uses each framework's verb: `mem0.add` / `graphiti.add_episode` / `cognee.cognify` / `rag.upsert` / `full_context.append`.

Every span carries `langfuse.session.id` so the three traces above always group into one session. Every CSV row carries `session_id` and `trace_id` columns so a row deep-links straight to its trace; the session_id format is `{prefix}{memory}_conv{N}_{mode}_{run_id}` where `run_id = uuid.uuid4().hex[:8]` is generated once per `/experiment` call. Pass `session_id` on the `/experiment` body to resume into an existing run; otherwise resume is keyed on session_id, so re-running with a fresh call starts clean instead of inheriting the previous run's CSV state.

### Data flow

```
make experiment ──▶ benchmark ──▶ /experiment
                          │
                          ├──▶ verifies toxics on toxiproxy (412 on mismatch)
                          ├──▶ for each message: coordinator /memorize  ──▶  memory backend
                          ├──▶ asks coordinator /ask  ──▶  ollama-tool-calling
                          │                                 │
                          │                                 ├──▶ memory /remember (toxiproxy)
                          │                                 └──▶ responder /respond (toxiproxy)
                          │                                          │
                          │                                          └──▶ litellm ──▶ OpenAI
                          ├──▶ LLM-as-judge panel (LLM_AS_JUDGE_SEED parallel calls,
                          │     majority-voted) — gpt-4o-mini via litellm
                          ├──▶ direct cgroup + litellm /metrics snapshot
                          │     (CPU/RAM/disk/net + tokens/cost + responder_context_tokens)
                          └──▶ appends one row per persisted memory and per Q&A to
                              {prefix}{backend}_{mode}.csv
```

`phase` is `warmup`, `load`, or `ask`.
- `warmup` — one row per `(memory, conv, mode)` leg, written immediately after the pre-leg reset. Captures one-time backend init (Graphiti `build_indices_and_constraints`, Neo4j fulltext-index population, Qdrant collection creation) so it doesn't get folded into row #1 of the load.
- `load` — one row per persisted message. `seed` = session number, `question` = global message counter, `category` is null.
- `ask` — one row per question. `seed` is null. `category` = LoCoMo question category (1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial). `judge` = majority-vote verdict from `judge_n` independent judge calls; `judge_correct_votes` and `judge_labels` preserve the raw ballots.

`wall_ms` is split into `compute_ms` (request-level latency a production system would pay) and `flush_ms` (bench-side I/O quiescence wait, instrumentation artifact). `experiment_id` is a stable hash over `(memory, conv, name_prefix, toxic_latency, toxic_jitter, toxic_bandwidth)`, shared across phases / questions of the same configuration. `experiment_name` carries the human-readable form `{prefix}{backend}_conv{conv}_{mode}`.

## Quick start

### Prerequisites

- Docker + Docker Compose
- GNU Make
- An OpenAI API key
- ~12 GB RAM (Ollama + Neo4j + Qdrant)

### First run

```bash
cp .env.example .env
# edit .env, set OPENAI_API_KEY=sk-...

make build       # builds all images
make start       # brings the full stack up; pk/sk auto-generated on first run

# smoke test (single CONV, both regimes, 1 judge per answer, 3 questions per category)
make experiment-test CONV=0

# full publishable sweep (all 10 LOCOMO convs × both regimes × 5 backends, 3 judges per answer)
make experiment
```

### Smoketests

Three preset depths, smallest to largest. All three sweep both regimes across every backend in `BACKENDS`, with `LLM_AS_JUDGE_SEED=1` and `KEEP_STATE=1` so the constrained leg reuses the unconstrained load.

```bash
make experiment-test-s CONV=0   # 5 messages, 1 question of category 2 — fastest, ~minutes
make experiment-test   CONV=0   # 119 messages, 3 questions × cats 1-4 — calibration depth
make experiment-test-l CONV=0   # 199 messages, 3 questions × cats 1-4 — deeper smoke
```

Narrow with e.g. `BACKENDS="mem0 graphiti"`. See `make experiment-test*` in the targets table for the full knob list.

### Make targets

| Command                | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `make build`           | Rebuild every image from scratch (`--no-cache --pull`). Aborts if `OPENAI_API_KEY` is missing from `.env`. Always re-resolves `LANGFUSE_PUBLIC_URL` from EC2 IMDSv2 (or keeps the existing value off-EC2) and echoes the resolved URL on completion so you can confirm what got baked into the env. |
| `make start`           | Bring the full stack up headlessly. On first run, `_bootstrap_env_file` auto-generates `LANGFUSE_PUBLIC_KEY/SECRET_KEY` (`pk-lf-…`/`sk-lf-…`), `LANGFUSE_OTEL_BASIC_AUTH`, and (on EC2) `LANGFUSE_PUBLIC_URL` from instance metadata. Langfuse v3 is bootstrapped via `LANGFUSE_INIT_*` against those keys — no UI dance. |
| `make stop`            | Stop containers; volumes preserved.                              |
| `make clean`           | Stop, then drop only the memory volumes (`qdrant-data`, `neo4j-data`, `neo4j-logs`). Langfuse history and ollama models stay. |
| `make reset`           | Stop, then drop **every** named volume + the dmas-network. Next `make start` starts blank. |
| `make experiment`      | Full sweep — every conv in `CONVS` × {unconstrained, constrained} × every backend in `BACKENDS`. No `KEEP_STATE`, every leg cleans state, loads the conv in full, asks every non-adversarial question once. Each answer is graded by `LLM_AS_JUDGE_SEED` parallel judge calls and majority-voted. Defaults: `CONVS="0 1 2 3 4 5 6 7 8 9"`, `BACKENDS="mem0 graphiti rag cognee full_context"`, `LLM_AS_JUDGE_SEED=3`. Override e.g. `make experiment BACKENDS="mem0 graphiti" CONVS="0 5"`. |
| `make experiment-leg`  | Single-leg primitive: one `(CONV, MODE)` × backends. Used by the sweeps; rarely called directly. Knobs: `CONV=N MODE=… LLM_AS_JUDGE_SEED=N BACKENDS="…" QUESTIONS=N MESSAGES=N QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=N KEEP_STATE=1 NAME_PREFIX=…`. |
| `make experiment-test-s` | **Short smoke**: both regimes for one CONV (default 0) with `MESSAGES=5 QUESTION_TYPES=2 Q_PER_TYPE=1 LLM_AS_JUDGE_SEED=1 KEEP_STATE=1 NAME_PREFIX=test_s_`. Five messages is the smallest `CONV=0` prefix that covers the first multi-hop question's evidence — runs in minutes. |
| `make experiment-test` | **Calibration smoke**: `MESSAGES=119 QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=3 LLM_AS_JUDGE_SEED=1 KEEP_STATE=1 NAME_PREFIX=test_`. **`MESSAGES=119` is hand-picked** — smallest `CONV=0` prefix that covers the evidence for the first 3 questions in each non-adversarial category. `KEEP_STATE=1` means subsequent invocations of the same `(backend, mode)` re-ask without reloading — fast iteration on retrieval / responder / judge. |
| `make experiment-test-l` | **Long smoke**: `MESSAGES=199 QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=3 LLM_AS_JUDGE_SEED=1 KEEP_STATE=1 NAME_PREFIX=test_l_`. Same shape as `experiment-test` but loads more messages so retrieval has more haystack to work against — useful when calibration shows accuracy saturating at the 119-message prefix. |
| `make experiments`     | Alias of `make experiment` (back-compat).                        |
| `make logs` / `ps`     | Tail logs / list containers.                                     |

### Per-row resume

The benchmark dedupes per CSV row, so any failure that leaves the per-experiment file on disk is recoverable.

- **Load phase** — `loaded_messages()` scans the per-experiment file for `phase=load` rows matching `(backend, conv, name_prefix, session_id)` and skips any global-message-counter already present. `ok`, `failed`, and `skipped` rows all count as "done"; to retry a failed message, delete its row from the CSV first.
- **Ask phase** — `already_done()` skips any `(conv, question)` already present as a `phase=ask` row in the same file.
- **Toxic verification** — every `/ask` re-checks toxiproxy state against the requested mode (`verify_all`); a mid-run drift aborts with HTTP 412 instead of silently mis-recording.
- **Pre-leg reset is resume-aware** — when `loaded_messages()` already returns rows for the current `session_id`, the bench skips the wipe + warmup steps and continues straight into the load loop. The first ever leg of a config still pays warmup once.
- **Session-keyed resume vs. fresh runs** — `session_id` is `{prefix}{backend}_conv{N}_{mode}_{run_id}` where `run_id = uuid.uuid4().hex[:8]` is generated **per `/experiment` call**. A fresh `make experiment …` mints a new `run_id`, so prior CSV rows under different session_ids are NOT skipped — the run starts clean. To deliberately resume into an existing run, pass that run's `session_id` on the `/experiment` body.

How to resume: kill any run with Ctrl-C and re-issue the same `make` command. The next call rebuilds resume state from the CSVs on disk; per-experiment files are partitioned by `(name_prefix, backend, mode)` (filename `{prefix}{backend}_{mode}.csv`), and convs share a file, distinguished by the `conversation_index` column. If a leg got stuck mid-load with the memory backend in a half-loaded state, the next call sees the existing `phase=load` rows and resumes from message N+1 against whatever the backend already persisted; for that to work, `KEEP_STATE=1` (smoke targets default to this) or the operator must not run `make clean` between the failure and the retry — `make clean` drops Qdrant + Neo4j volumes, after which the CSV says "loaded" but the backend is empty.

## Configuration

### Environment variables

| Variable                 | Description                                                                | Default                          |
| ------------------------ | -------------------------------------------------------------------------- | -------------------------------- |
| `OPENAI_API_KEY`         | Real OpenAI key — used only by litellm; agents see `sk-litellm-master`.    | (required, set in `.env`)        |
| `LLM_MODEL`              | Cloud chat/completion model — used by responder, judge, and the mem0/graphiti/cognee extractors. | `gpt-4o-mini`                    |
| `OLLAMA_MODEL`           | Local SLM the coordinator calls via litellm.                               | `qwen2.5:3b-instruct-q4_K_M`     |
| `RAG_EMBED_MODEL`        | Embedding model for vector retrieval (mem0/graphiti/cognee/rag).           | `text-embedding-3-small`         |
| `MEMORIES_SEARCH_LIMIT`  | Retrieval `top_k` for mem0, graphiti, cognee, rag.                         | `20`                             |
| `LLM_AS_JUDGE_SEED`      | Number of independent judge calls per answer; majority-voted.              | `3` (smoke uses `1`)             |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Self-hosted Langfuse v3 project keys, pre-seeded into the `dmas` project via `LANGFUSE_INIT_PROJECT_*`. | auto-generated on first `make start` |
| `LANGFUSE_OTEL_BASIC_AUTH` | base64(public:secret) — read by litellm's OTel exporter as the `Authorization: Basic …` header. | auto-derived on first `make start` |
| `LANGFUSE_PUBLIC_URL`    | Browser-facing URL of langfuse-web; used to build the `trace_url` column in CSVs. | EC2 public IP (auto-detected via IMDSv2) or `http://localhost:3000` |

### Network fault injection

Toxiproxy proxies sit in front of `memory` and `responder`. **Set toxics on the toxiproxy admin API before running `make experiment`** — the benchmark verifies them against the request body and rejects with HTTP 412 on mismatch (`dmas/benchmark/app/toxics.py`). The three request fields the benchmark checks:

- `latency` (ms)
- `jitter` (ms)
- `bandwidth` (KB/s)

Per `MODE`: `unconstrained` clears all toxics; `constrained` applies `CONSTRAINED_LATENCY` / `CONSTRAINED_JITTER` / `CONSTRAINED_BANDWIDTH` (defaults 100ms / 20ms / 512 KB/s).

## Project structure

```
dmas-memory/
├── dmas/
│   ├── benchmark/        # Experiment runner: /experiment (drives /memorize per message + ASK loop), judges, metrics, per-experiment CSV writer
│   ├── coordinator/      # Slim /ask handler (Ollama tool-calling)
│   ├── memory/           # mem0 + Graphiti + Cognee + RAG + FullContext (per-request backend selection)
│   ├── responder/        # Final-answer generator
│   ├── shared/           # otel_init.py, litellm_usage.py, models.py — copied into every service image; one canonical implementation
│   ├── litellm/config.yaml   # Single LLM gateway: gpt-4o-mini + qwen2.5:3b + text-embedding-3-small (no catch-all)
│   └── docker-compose.yml    # Build context is `dmas/` so each service image can COPY shared/
├── experiments/
│   ├── experiments.sh    # Thin wrapper around `make experiment`
│   ├── results.ipynb     # Statistical analysis (§§1–9b) — every chapter renders a table followed by a bar chart with a constrained-vs-unconstrained Δ panel
│   └── results/          # Per-experiment CSVs ({prefix}{backend}_{mode}.csv)
├── Makefile
└── .env.example
```

## Analysis

`experiments/results.ipynb` reproduces the paper's analysis end-to-end by globbing every `*.csv` under `experiments/results/` and concatenating into a single DataFrame. **Every chapter prints a table and then renders a bar chart** with a `constrained − unconstrained` Δ panel underneath, so the regime impact per backend is always visible at a glance:

1. **Financial cost** — LLM tokens & USD per `memory × regime × phase`, split into edge (ollama) vs cloud (OpenAI). One bar chart per phase.
2. **Computational cost** — CPU / RAM / disk / network split cloud vs. edge. One bar chart per resource axis.
2b. **Responder context length per question** — mean / median / p95 of `responder_context_window_tokens` (the actual `prompt_tokens` the answering LLM consumed), plus `tokens_per_correct` as an efficiency metric, with bar charts for each.
3. **Temporal cost** — load / ask wall-clock seconds, bars stacked by `compute_ms` (production cost) vs `flush_ms` (bench artifact).
4. **Response distribution** — CORRECT / WRONG / UNKNOWN per `(memory, regime)`. **All memories rendered side-by-side in a single chart** with two clustered bars per memory (unconstrained labelled `U`, constrained labelled `C` and drawn with a hatch + red border) so every backend is directly comparable.
5. **Wilson 95% CIs** for accuracy — bars with CI error bars per memory × regime.
6. **Two-proportion z-tests** — every pair of memories within each regime, plus each memory's own constrained-vs-unconstrained gap. Significant pairs (p < α=0.05) cross the dashed reference line in the `-log₁₀(p)` chart. (No longer hard-coded to the mem0/graphiti pair, so prefix-per-backend CSV layouts work.)
6b. **LLM-as-judge agreement** — `judge_n` panel size and the majority-correct rate per `(memory, regime)`. With the current schema, only `judge_n` and the majority `judge_verdict` are persisted, so the majority-correct rate equals the §5 accuracy and is plotted as cross-check rather than independent agreement.
7. **TCO** — per-`(experiment, memory)` cost stacked by resource type (cpu / ram / disk / net / llm) with AWS Fargate pricing.
8. **Statistical Pareto efficiency** — pairwise dominance check across **every** pair of memories present in the data (not only mem0 vs graphiti); declares one dominant only if cheaper *and* the accuracy gap is not significant. Followed by a TCO bar chart per memory × regime with a Δ panel.
9. **Accuracy by question category** — per-memory grouped bar with regime as hue and a Δ panel; cat 5 stays visible here.
9b. **Accuracy table by LoCoMo category** — pivot with rows = `(memory, regime)`, columns = LoCoMo categories, cells `k/n (acc%)`, plus an `overall (1-4)` j-score column. The j-score is also rendered as a final bar chart.

### Methodology

- **LoCoMo j-score** — accuracy is computed on categories 1–4 only; cat 5 (adversarial) is excluded by default in §§1–§8 (settled in [zep-papers#5](https://github.com/getzep/zep-papers/issues/5)). The `/experiment` endpoint filters cat 5 by default (`include_adversarial=false`).
- **Identical responder prompt across backends** — we don't fork the responder system prompt by backend, even when one backend's authors prefer a different phrasing. mem0/Zep's dispute settled in favour of "uniform prompt across baselines"; we follow that.
- **Single ask, multi-judge** — each question is asked **once**. The LLM judge runs `LLM_AS_JUDGE_SEED` independent times (default 3) and the per-row `judge_verdict` column is the majority vote (>50% CORRECT ⇒ CORRECT). The CSV persists the panel size (`judge_n`), the majority verdict (`judge_verdict`), and the reasoning of the first judge call (`judge_reason`); per-call labels and correct-vote counts are not written to keep rows compact.
- **Determinism** — every LLM call in the system is at `temperature=0`, including extraction (mem0/graphiti/cognee), responder, judge, and the graphiti cross-encoder reranker.
- **Parallel Graphiti search** — `dmas/memory/app/services/graphiti_service.py:remember_async` runs the edge (facts) and node (entity summaries) searches in parallel via `asyncio.gather`, mirroring Zep's corrected `zep_locomo_search.py`.
- **Verbatim ingestion contract** — every backend ingests one message per write, in chronological session order, with the same per-call telemetry envelope. mem0 and Graphiti use loading and retrieval logic exactly as their authors run it (`mem0ai/memory-benchmarks/benchmarks/locomo`, `getzep/zep-papers/.../zep_locomo_ingestion.py`); Cognee is driven per-message even though its authors normally batch-cognify, to keep the per-row `wall_ms` / token / cost numbers comparable across backends (documented in `cognee_service.py`).

### Time anchoring

Each backend gets the LOCOMO session timestamp through whichever channel its API accepts:

| Backend  | Channel                                                            |
| -------- | ------------------------------------------------------------------ |
| mem0     | `[ISO 8601 UTC]` prefix in message `content` (OSS API has no time kwarg) |
| RAG      | `[ISO 8601 UTC]` prefix in stored text (no extractor; symmetric with the others) |
| Cognee   | `[ISO 8601 UTC]` prefix in episode text (cognee 0.5.6 has no time kwarg) |
| Graphiti | `reference_time` kwarg on `add_episode` (graphiti's bi-temporal anchor — appending into the body would reproduce the Zep-flagged "improper timestamp handling") |

### LLM-as-a-judge

The judge prompt is verbatim from Zep's `locomo_grader` (`getzep/zep-papers, kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py`) and asks for a JSON object with `label` (CORRECT|WRONG) and a one-sentence `reasoning`. `judge_majority(question, gold, answer, n)` runs the call `n` times and majority-votes. The CSV persists the consensus (`judge_verdict`), the panel size (`judge_n`), and the first call's reasoning (`judge_reason`). Unanimous non-CORRECT/non-WRONG verdicts (e.g. all-PLACEHOLDER from an unimplemented judge or all-ERROR from a broken one) are surfaced as that label rather than masked as WRONG; otherwise non-CORRECT votes count as WRONG for the tally. Rows where the `/ask` call itself errored skip the judge — `judge_verdict`, `judge_reason`, `judge_n` are left null and the failure is recorded in `error` instead of being silently labeled WRONG.

### Resource-tracking

The bench reads kernel pseudo-files directly — no TSDB middleman, no scrape interval rounding:

- **CPU / disk / network** — `/sys/fs/cgroup/.../{cpu.stat, io.stat}` + `/proc/<container_pid>/net/dev`. Network is **tx-only** so each byte is counted once at its sender; toxiproxy is excluded from tx aggregation because its tx is just retransmit of upstream bytes. `dmas/benchmark/app/cgroup_metrics.py` maps each container_id → `group=edge|cloud` via the docker socket and sums per group.
- **RAM** — diff of `memory.peak` between two snapshots. Captures the *additional* working-set high-water mark induced by the call (non-negative, monotonic). Stored as `ram_*_peak_bytes` so consumers don't mistake it for a `memory.current` average.
- **Disk attribution under async DBs** — Neo4j and Qdrant both checkpoint asynchronously; without intervention, per-call `disk_cloud_bytes` would understate Graphiti's and Cognee's writes. The bench (a) sets Neo4j `db.checkpoint.interval.tx=1` so it checkpoints after every transaction, (b) blocks the t1 cgroup snapshot until **cloud-group disk I/O quiesces** (`wait_io_quiet` in `cgroup_metrics.py`, group-wide aggregate so Qdrant is included too), and (c) reports the added wait as `flush_ms` so it's distinguishable from `compute_ms` (the production-equivalent latency). Edge group is excluded from the quiescence check because Ollama doesn't flush asynchronously.
- **Tokens / cost** — `litellm:4000/metrics`, parsed live before+after each call. Split by the `model` label into edge (ollama via the `local-slm` alias → `qwen2.5:3b-instruct-q4_K_M`, free in litellm pricing) vs cloud (OpenAI passthrough). LiteLLM does the per-model price lookup; no pricing JSON to maintain.
- **Responder context length** — the `prompt_tokens` of the OpenAI completion that produced the final answer is captured in `responder_context_tokens`, so analyses can normalise accuracy by how much retrieved context the responder had to process per question.

Each row reflects the actual call window — even sub-second calls get real per-row deltas.

## References

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

## License

See [LICENSE.txt](LICENSE.txt) for details.
