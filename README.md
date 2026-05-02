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

The bench reads resource counters directly from kernel pseudo-files — no monitoring stack required:
- **`/sys/fs/cgroup`** (mounted ro) — per-container CPU (`cpu.stat`), RAM (`memory.peak`), disk (`io.stat`).
- **`/proc/<pid>/net/dev`** (host `/proc` mounted ro) — per-namespace network counters; tx-only, toxiproxy excluded so each byte is counted once at its sender.
- **`/var/run/docker.sock`** (mounted ro) — one-shot container label/PID lookup, cached.
- **`litellm:4000/metrics`** — live tokens/cost counters, split by model name into edge (ollama-served) vs cloud (OpenAI passthrough).

No Prometheus, no Telegraf, no scrape interval — every snapshot reflects the moment of the read.

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
make setup       # one-time: brings Langfuse up, prompts for keys, persists to .env
make start       # brings the rest of the stack up

# smoke test (single CONV, both regimes, 1 judge per answer, 3 questions per category)
make experiment-test CONV=0

# full publishable sweep (all 10 LOCOMO convs × both regimes × 5 backends, 3 judges per answer)
make experiment
```

### Make targets

| Command                | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `make build`           | Rebuild every image from scratch (`--no-cache --pull`). Aborts if `OPENAI_API_KEY` is missing from `.env`. |
| `make setup`           | One-time: bring Langfuse up, prompt for `pk-lf-…` / `sk-lf-…`, persist to `.env`. Required before the first `make start`. |
| `make start`           | Bring the rest of the stack up. Requires Langfuse keys in `.env` (run `make setup` first). |
| `make stop`            | Stop containers; volumes preserved.                              |
| `make clean`           | Stop, then drop only the memory volumes (`qdrant-data`, `neo4j-data`, `neo4j-logs`). Langfuse history and ollama models stay. |
| `make reset`           | Stop, then drop **every** named volume + the dmas-network. Next `make setup` starts blank. |
| `make experiment`      | Full sweep — every conv in `CONVS` × {unconstrained, constrained} × every backend in `BACKENDS`. No `KEEP_STATE`, every leg cleans state, loads the conv in full, asks every non-adversarial question once. Each answer is graded by `LLM_AS_JUDGE_SEED` parallel judge calls and majority-voted. Defaults: `CONVS="0 1 2 3 4 5 6 7 8 9"`, `BACKENDS="mem0 graphiti rag cognee full_context"`, `LLM_AS_JUDGE_SEED=3`. Override e.g. `make experiment BACKENDS="mem0 graphiti" CONVS="0 5"`. |
| `make experiment-leg`  | Single-leg primitive: one `(CONV, MODE)` × backends. Used by the sweeps; rarely called directly. Knobs: `CONV=N MODE=… LLM_AS_JUDGE_SEED=N BACKENDS="…" QUESTIONS=N MESSAGES=N QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=N KEEP_STATE=1 NAME_PREFIX=…`. |
| `make experiment-test` | Smoke: both regimes for one CONV with `MESSAGES=119 QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=3 LLM_AS_JUDGE_SEED=1 KEEP_STATE=1 NAME_PREFIX=test_`. **`MESSAGES=119` is hand-picked**: smallest `CONV=0` prefix that covers the evidence for the first 3 questions in each non-adversarial category. `KEEP_STATE=1` means subsequent invocations of the same `(backend, mode)` re-ask without reloading — fast iteration on retrieval / responder / judge. |
| `make experiments`     | Alias of `make experiment` (back-compat).                        |
| `make logs` / `ps`     | Tail logs / list containers.                                     |

### Per-row resume

The benchmark dedupes by `(backend, conv, name_prefix, toxic_*)` for the load phase (per-message, by global counter) and by `(conv, question)` for ASK. Kill any run with Ctrl-C and re-issue the same `make` command — it picks up where it stopped. Per-experiment files are partitioned by `(name_prefix, backend, mode)` (filename `{prefix}{backend}_{mode}.csv`); convs share a file, distinguished by the `conversation_index` column.

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
| `LANGFUSE_*`             | Self-hosted Langfuse keys (auto-generated by `make setup`).                | populated by `make setup`        |

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
│   ├── litellm/config.yaml   # Single LLM gateway: gpt-4o-mini + qwen2.5:3b + text-embedding-3-small (no catch-all)
│   └── docker-compose.yml
├── experiments/
│   ├── experiments.sh    # Thin wrapper around `make experiment`
│   ├── results.ipynb     # Statistical analysis (§§1–9b)
│   └── results/          # Per-experiment CSVs ({prefix}{backend}_{mode}.csv)
├── Makefile
└── .env.example
```

## Analysis

`experiments/results.ipynb` reproduces the paper's analysis end-to-end by globbing every `*.csv` under `experiments/results/` and concatenating into a single DataFrame:

1. **Financial cost** — LLM tokens & USD per `memory × regime × phase`, split into edge (ollama) vs cloud (OpenAI).
2. **Computational cost** — CPU / RAM / disk / network split cloud vs. edge.
2b. **Responder context length per question** — mean / median / p95 of `responder_context_tokens` (the actual `prompt_tokens` the answering LLM consumed), plus `tokens_per_correct` as an efficiency metric.
3. **Temporal cost** — load / ask wall-clock seconds (`compute_ms` vs `flush_ms` separated).
4. **Response distribution** — CORRECT / WRONG / UNKNOWN per regime, plus a bar chart.
5. **Wilson 95% CIs** for accuracy.
6. **Two-proportion z-tests** for pairwise comparisons.
6b. **LLM-as-judge agreement** — mean `judge_correct_votes / judge_n` and majority-correct rate per `(memory, regime)`.
7. **TCO** — five linear-scale diagrams (CPU, RAM, Disk, Network, Tokens) with AWS Fargate pricing.
8. **Statistical Pareto efficiency** — declares one backend dominant only if cheaper *and* the accuracy gap is not significant.
9. **Accuracy by question category × judge** (cat 5 visible only here).
9b. **Accuracy table by LoCoMo category** — pivot with rows = `(memory, regime)`, columns = LoCoMo categories, cells `k/n (acc%)`, plus an `overall (1-4)` j-score column.

### Methodology

- **LoCoMo j-score** — accuracy is computed on categories 1–4 only; cat 5 (adversarial) is excluded by default in §§1–§8 (settled in [zep-papers#5](https://github.com/getzep/zep-papers/issues/5)). The `/experiment` endpoint filters cat 5 by default (`include_adversarial=false`).
- **Identical responder prompt across backends** — we don't fork the responder system prompt by backend, even when one backend's authors prefer a different phrasing. mem0/Zep's dispute settled in favour of "uniform prompt across baselines"; we follow that.
- **Single ask, multi-judge** — each question is asked **once**. The LLM judge runs `LLM_AS_JUDGE_SEED` independent times (default 3) and the per-row `judge` column is the majority vote (>50% CORRECT ⇒ CORRECT). `judge_correct_votes / judge_n` and `judge_labels` preserve the raw ballots.
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

The judge prompt is verbatim from Zep's `locomo_grader` (`getzep/zep-papers, kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py`) and asks for a JSON object with `label` (CORRECT|WRONG) and a one-sentence `reasoning`. `judge_majority(question, gold, answer, session_date, n)` runs the call `n` times and majority-votes; the `judge` column is the consensus, `judge_correct_votes / judge_n` is the agreement fraction, and `judge_labels` is the pipe-separated ballot. Rows where the `/ask` call itself errored skip the judge — `judge`, `judge_reason`, `judge_n` are left null and the failure is recorded in `error` instead of being silently labeled WRONG.

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
