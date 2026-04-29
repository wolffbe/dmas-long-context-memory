# Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems

A distributed multi-agent system testbed for benchmarking vector-based vs. graph-based long-term conversational memory in different network scenarios.

## Overview

This project compares two approaches to persistent memory in LLM-based multi-agent systems:

| Approach          | Backend                                        | Storage | Search Method            |
| ----------------- | ---------------------------------------------- | ------- | ------------------------ |
| **Vector Memory** | [mem0](https://github.com/mem0ai/mem0)         | Qdrant  | Semantic similarity      |
| **Graph Memory**  | [Graphiti](https://github.com/getzep/graphiti) | Neo4j   | Hybrid edge search       |
| **Classical RAG** | in-house (`RagService`)                        | Qdrant  | Top-k cosine over individual turns |
| **Cognee Memory** | [cognee](https://github.com/topoteretes/cognee) | Neo4j + Qdrant | Knowledge-graph completion (LLM-extracted entities/edges) |
| **Full Context**  | in-house (`FullContextService`)                | in-process | None — full conversation JSON to responder |

Memory loading and retrieval mirror the upstream evaluation harnesses verbatim — `mem0ai/memory-benchmarks` for mem0, `getzep/zep-papers` for Graphiti — so this testbed reproduces what each project's authors do, then layers a distributed-systems / cost-tracking framework on top.

The system evaluates memory retrieval on the [LOCOMO benchmark](https://github.com/snap-research/locomo) — a dataset for very long-term conversational memory in LLM agents.

## Research Context

This repository is the official implementation of a study evaluating long-term memory frameworks in Distributed Multi-Agent Systems (DMAS).

[![arXiv](https://img.shields.io/badge/arXiv-2601.07978-b31b1b.svg)](https://arxiv.org/abs/2601.07978)

While DMAS leverage Large Language Models (LLMs) for collaborative intelligence, systematic evaluations of their memory under network constraints are often lacking. This project addresses that gap by comparing **mem0** (vector-based) and **Graphiti** (graph-based) on the **LOCOMO** long-context benchmark across unconstrained and constrained network regimes.

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

The stack is split across two Docker Compose files:

### `dmas/` — application services + LLM gateway

| Service         | Port      | Group | Description                                                                                            |
| --------------- | --------- | ----- | ------------------------------------------------------------------------------------------------------ |
| **benchmark**   | 8002      | cloud | Owns LOCOMO loading, the Q&A loop, judges, metric capture, and CSV writes.                              |
| **coordinator** | 8003      | edge  | Slim per-question handler — Ollama tool-calling → search\_memory → answer.                              |
| **memory**      | (8005 via toxiproxy) | cloud | mem0, Graphiti, RAG, FullContext all instantiated; backend chosen per request.                  |
| **responder**   | (8006 via toxiproxy) | cloud | Final-answer generator (OpenAI gpt-4o-mini by default).                                          |
| **ollama**      | 11435     | edge  | Local SLM inference for the coordinator.                                                                |
| **qdrant**      | 6333      | cloud | Vector store for mem0 and RAG.                                                                          |
| **neo4j**       | 7474/7687 | cloud | Graph store for Graphiti. Configured with `db.checkpoint.interval.tx=1` so per-call disk attribution is honest. |
| **litellm**     | 4000      | cloud | Single OpenAI-compatible proxy for both OpenAI and Ollama; emits Prometheus metrics.                    |
| **toxiproxy**   | 8474      | cloud | One container, two named proxies (memory / responder). Toxics set externally; benchmark verifies. |

### Observability

The bench reads resource counters directly from kernel pseudo-files — no monitoring stack required:
- **`/sys/fs/cgroup`** (mounted ro) — per-container CPU (`cpu.stat`), RAM (`memory.current`), disk (`io.stat`).
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
                          ├──▶ judges (mem0, zep, dmas) — OpenAI via litellm
                          ├──▶ direct cgroup + litellm /metrics snapshot (CPU/RAM/disk/net + tokens/cost)
                          └──▶ appends one row per persisted memory and per Q&A to results.csv
```

`phase` is `warmup`, `load`, or `ask`.
- `warmup` — one row per `(memory, conv, mode)` leg, written immediately after the pre-leg reset. Captures one-time backend init (graphiti `build_indices_and_constraints`, qdrant collection creation) so it doesn't get folded into row #1 of the load.
- `load` — one row per persisted message. `seed` = session number, `question` = global message counter, `category` is null.
- `ask` — one row per `(seed, question)`. `category` = LoCoMo question category (1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial).

`wall_ms` is split into `compute_ms` (request-level latency a production system would pay) and `flush_ms` (bench-side I/O quiescence wait, instrumentation artifact). `experiment_id` is a stable hash over `(memory, conv, toxic_latency, toxic_jitter, toxic_bandwidth)`, shared across phases / seeds / questions.

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
make start       # brings the stack up

# run an experiment (load + ask in one call, per backend, both regimes)
make experiment CONV=0 SEEDS=1
make experiments  # full grid: both modes for one CONV across all backends
```

`MODE=constrained` swaps in toxiproxy latency/jitter/bandwidth; the benchmark sets the toxics itself based on `MODE`.

### Make targets

| Command                | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `make build`           | Rebuild every image from scratch (`--no-cache --pull`). Aborts if `OPENAI_API_KEY` is missing from `.env`. |
| `make setup`           | One-time: bring Langfuse up, prompt for `pk-lf-…` / `sk-lf-…`, persist to `.env`. Required before the first `make start`. |
| `make start`           | Bring the rest of the stack up. Requires Langfuse keys in `.env` (run `make setup` first). |
| `make stop`            | Stop containers; volumes preserved.                              |
| `make clean`           | Stop, then drop only the memory volumes (`qdrant-data`, `neo4j-data`, `neo4j-logs`). Langfuse history and ollama models stay. |
| `make reset`           | Stop, then drop **every** named volume + the dmas-network. Next `make setup` starts blank. |
| `make experiment`      | `CONV=N [MODE=…] [SEEDS=N] [BACKENDS="…"] [LIMIT=N] [LOAD_LIMIT=N] [QUESTION_TYPES=1,2,3,4] [Q_PER_TYPE=N] [KEEP_STATE=1] [NAME_PREFIX=…]` — bench loads CONV per-message into each backend, runs the Q&A loop, streams NDJSON. Rows land in `experiments/results/results.csv`. `QUESTION_TYPES` filters to LoCoMo categories; `Q_PER_TYPE` keeps the first N questions per category in qa-list order; `KEEP_STATE=1` skips the post-leg memory wipe so the next call with the same `NAME_PREFIX` reuses the load. |
| `make experiment-test` | Calibration-style smoke: `MESSAGES=119 QUESTION_TYPES=1,2,3,4 Q_PER_TYPE=3 SEEDS=1 KEEP_STATE=1 NAME_PREFIX=test_`, unconstrained mode. **`MESSAGES=119` is hand-picked**: that's the smallest `CONV=0` prefix the operator has verified covers the evidence for the first 3 questions in each non-adversarial category. The benchmark does not auto-derive it — change it (or `TEST_MESSAGES`) for any other CONV. First run pays the load + ask cost; subsequent runs skip the load (resume marker) and only re-ask, so iterating on retrieval / responder / judge is fast. |
| `make experiments`     | Run the full grid via `experiments/experiments.sh` (both modes for one CONV across all backends). |
| `make logs` / `ps`     | Tail logs / list containers.                                     |

### Per-row resume

The benchmark dedupes by `(memory, conv, toxic_*)` for the load phase (per-message, by global counter) and additionally by `(seed, question)` for ASK. Kill any run with Ctrl-C and re-issue the same `make` command — it picks up where it stopped.

> **Note on schema changes:** the load row format (`question = global counter`, `seed = session number`) was set in this revision. CSVs from earlier code that still hold rows like `question="session_K:N"` aren't matched by the new resume key and will be silently re-loaded into duplicate qdrant/neo4j entries. Wipe `experiments/results/results.csv` to header-only when migrating an old run.

## Configuration

### Environment variables

| Variable                                        | Description                                                       | Source                          |
| ----------------------------------------------- | ----------------------------------------------------------------- | ------------------------------- |
| `OPENAI_API_KEY`                                | Real OpenAI key — used only by litellm; agents see `sk-litellm-master`. | `.env`                          |
| `OLLAMA_MODEL`                                  | Local SLM the coordinator calls via litellm.                      | `.env`                          |

### Network fault injection

Toxiproxy proxies sit in front of `memory` and `responder`. **Set toxics on the toxiproxy admin API before running `make experiment`** — the benchmark verifies them against the request body and rejects with HTTP 412 on mismatch (`dmas/benchmark/app/toxics.py`). The four request fields the benchmark checks:

- `latency` (ms)
- `jitter` (ms)
- `bandwidth` (KB/s)

## Project structure

```
dmas-memory/
├── dmas/
│   ├── benchmark/        # Experiment runner: /experiment (drives /memorize per message + ASK loop), judges, metrics, CSV writer
│   ├── coordinator/      # Slim /ask handler (Ollama tool-calling)
│   ├── memory/           # mem0 + Graphiti (per-request backend selection)
│   ├── responder/        # Final-answer generator
│   ├── litellm/config.yaml   # One LLM gateway for OpenAI + Ollama
│   └── docker-compose.yml
├── experiments/
│   ├── experiments.sh    # Full grid runner (mem0 × graphiti × {unconstrained, constrained})
│   ├── results.ipynb     # Statistical analysis (Tables 1–6, Figures 2–3, Pareto)
│   └── results/results.csv
├── Makefile
└── .env.example
```

## Analysis

`experiments/results.ipynb` reproduces the paper's analysis end-to-end from `experiments/results/results.csv`:

1. **Financial cost** — LLM tokens & USD per `memory × regime × phase`, split into edge (ollama) vs cloud (OpenAI).
2. **Computational cost** — CPU / RAM / disk / network split cloud vs. edge.
3. **Temporal cost** — load / ask wall-clock minutes.
4. **Response distribution** — CORRECT / WRONG / IDK per regime, plus a bar chart.
5. **Wilson 95% CIs** for accuracy.
6. **Two-proportion z-tests** for all four pairwise comparisons.
6b. **Mean ± std-dev across seeds** (LoCoMo j-score protocol — average of 10 independent seeds).
7. **TCO** — five linear-scale diagrams (CPU, RAM, Disk, Network, Tokens) with AWS Fargate pricing.
8. **Statistical Pareto efficiency** — declares one backend dominant only if cheaper *and* the accuracy gap is not significant.
9. **Accuracy by question category × judge** (cat 5 visible only here).

### Methodology (after the mem0/Zep dispute, May 2025)

- **LoCoMo j-score** — accuracy is computed on categories 1–4 only; cat 5 (adversarial) is excluded by default in §1–§8 (settled in [zep-papers#5](https://github.com/getzep/zep-papers/issues/5)). The `/experiment` endpoint filters cat 5 by default (`include_adversarial=false`).
- **Identical responder prompt across backends** — we don't fork the responder system prompt by backend, even when one backend's authors prefer a different phrasing. mem0/Zep's dispute settled in favour of "uniform prompt across baselines"; we follow that.
- **Multi-seed averaging** — both teams converged on 10 independent seeds with mean ± std-dev. Default `SEEDS=3` for development; bump to `SEEDS=10` for the publishable number.
- **Parallel Graphiti search** — `dmas/memory/app/services/graphiti_service.py:remember_async` runs the edge (facts) and node (entity summaries) searches in parallel via `asyncio.gather`, mirroring Zep's corrected `zep_locomo_search.py`.
- **Verbatim ingestion** — both backends use loading and retrieval logic exactly as their authors run it (`mem0ai/memory-benchmarks/benchmarks/locomo`, `getzep/zep-papers/.../zep_locomo_ingestion.py`).

### LLM-as-a-judge

A single judge (`gpt-4o-mini` by default) runs on every successful Q&A row. The prompt is verbatim from Zep's `locomo_grader` (`getzep/zep-papers, kg_architecture_agent_memory/locomo_eval/zep_locomo_eval.py`) and asks for a JSON object with `label` (CORRECT|WRONG) and a one-sentence `reasoning`. Both fields are written to `judge` and `judge_reason` in `results.csv`. Rows where the `/ask` call itself errored skip the judge — `judge` and `judge_reason` are left null and the failure is recorded in the `error` column instead of being silently labeled WRONG.

### Resource-tracking

The bench reads kernel pseudo-files directly — no TSDB middleman, no scrape interval rounding:

- **CPU / disk / network** — `/sys/fs/cgroup/.../{cpu.stat, io.stat}` + `/proc/<container_pid>/net/dev`. Network is **tx-only** so each byte is counted once at its sender; toxiproxy is excluded from tx aggregation because its tx is just retransmit of upstream bytes. CPU and network are honest deltas. `dmas/benchmark/app/cgroup_metrics.py` maps each container_id → `group=edge|cloud` via the docker socket and sums per group.
- **RAM** — diff of `memory.peak` between two snapshots. Captures the *additional* working-set high-water mark induced by the call (non-negative, monotonic). Stored as `ram_*_peak_bytes` so consumers don't mistake it for a `memory.current` average.
- **Disk attribution under async DBs** — Neo4j checkpoints asynchronously; without intervention, per-call `disk_cloud_bytes` would understate graphiti's writes by 30 %+. The bench (a) sets `db.checkpoint.interval.tx=1` so Neo4j checkpoints after every transaction, and (b) blocks the t1 cgroup snapshot until cloud-side disk I/O quiesces (`wait_io_quiet` in `cgroup_metrics.py`). The added wait is reported as `flush_ms` so it's distinguishable from `compute_ms` (the production-equivalent latency).
- **Tokens / cost** — `litellm:4000/metrics`, parsed live before+after each call. Split by the `model` label into edge (ollama via the `local-slm` alias → `qwen2.5:3b-instruct-q4_K_M`, free in litellm pricing) vs cloud (OpenAI passthrough). LiteLLM does the per-model price lookup; no pricing JSON to maintain.

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
