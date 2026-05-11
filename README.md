# Cost and accuracy of long-term graph memory in distributed LLM-based multi-agent systems

[![arXiv](https://img.shields.io/badge/arXiv-2601.07978-b31b1b.svg)](https://arxiv.org/abs/2601.07978)

A distributed multi-agent system testbed for benchmarking five long-term conversational memory strategies under different network scenarios. **See the [arXiv paper](https://arxiv.org/abs/2601.07978) for the research questions, methodology, statistical Pareto analysis, and full results.** This README only covers what the repo contains and how to run it.

## Backends compared

| Approach          | Library                                                    | Storage         |
| ----------------- | ---------------------------------------------------------- | --------------- |
| **mem0**          | [mem0](https://github.com/mem0ai/mem0) v2.0.2              | Qdrant          |
| **Graphiti**      | [graphiti-core](https://github.com/getzep/graphiti) 0.29   | Neo4j           |
| **Cognee**        | [cognee](https://github.com/topoteretes/cognee) 1.0.9      | Neo4j + Qdrant  |
| **RAG**           | in-house `RagService`                                      | Qdrant          |
| **Full Context**  | in-house `FullContextService`                              | (in-process)    |

Loading and retrieval mirror the upstream evaluation harnesses verbatim (`mem0ai/memory-benchmarks`, `getzep/zep-papers`). Evaluation uses the [LOCOMO benchmark](https://github.com/snap-research/locomo).

## Quick start

Prereqs: Docker + Docker Compose, GNU Make, an OpenAI API key, ~12 GB RAM.

```bash
cd testbed
cp .env.example .env          # set OPENAI_API_KEY=sk-...
make build
make start

make experiment-test CONV=0   # smoke test
make experiment               # full publishable sweep
```

On Linux/macOS, `sudo bash install.sh` from `testbed/` installs Make + Docker.

### Smoke targets

| Target                    | Messages | Questions    | Duration |
| ------------------------- | -------- | ------------ | -------- |
| `make experiment-test-s`  | 5        | 1 (cat 2)    | minutes  |
| `make experiment-test`    | 119      | 3 × cats 1-4 | ~hour    |
| `make experiment-test-l`  | 199      | 3 × cats 1-4 | hours    |

All three sweep both network regimes for one `CONV` (default 0), reuse the load across regimes (`KEEP_STATE=1`), and run 1 judge call per answer. Narrow with `BACKENDS="mem0 graphiti"`.

### Make targets

| Command               | What it does                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `make build`          | Build every image from scratch. Aborts if `OPENAI_API_KEY` is missing.                      |
| `make start`          | Bring the full stack up. Langfuse pk/sk auto-generated on first run.                        |
| `make stop`           | Stop containers; volumes preserved.                                                          |
| `make clean`          | Stop + drop only memory volumes (`qdrant-data`, `neo4j-data`, `neo4j-logs`).                |
| `make reset`          | Stop + drop **every** named volume and the dmas-network.                                     |
| `make experiment`     | Full sweep: 10 LOCOMO convs × {unconstrained, constrained} × 5 backends × 3-judge majority. |
| `make experiment-leg` | Single `(CONV, MODE)` × backends. Knobs: `CONV MODE BACKENDS QUESTIONS MESSAGES Q_PER_TYPE QUESTION_TYPES KEEP_STATE`. |
| `make logs` / `ps`    | Tail logs / list containers.                                                                 |

## Network fault injection

Toxiproxy sits in front of `memory` and `responder`. The benchmark verifies the requested toxics against the live proxy state and rejects with HTTP 412 on mismatch (`dmas/benchmark/app/toxics.py`).

Two regimes: `unconstrained` clears all toxics; `constrained` applies `CONSTRAINED_LATENCY` / `CONSTRAINED_JITTER` / `CONSTRAINED_BANDWIDTH` (defaults 150 ms / 30 ms / 512 KB/s).

## Project structure

```
dmas-memory/
├── paper/                # Manuscript — published on arXiv: 2601.07978
└── testbed/              # Runnable benchmark — `cd testbed` before any `make`
    ├── dmas/
    │   ├── benchmark/        # /experiment endpoint, judges, metrics, CSV writer
    │   ├── coordinator/      # /ask handler with ollama tool-calling
    │   ├── memory/           # mem0 + Graphiti + Cognee + RAG + FullContext
    │   ├── responder/        # Final-answer generator
    │   ├── shared/           # otel_init.py, litellm_usage.py, models.py
    │   ├── litellm/config.yaml
    │   └── docker-compose.yml
    ├── experiments/
    │   ├── results.ipynb     # Reproduces every table and chart in the paper
    │   └── results/          # Per-experiment CSVs ({prefix}{backend}_{mode}.csv)
    ├── Makefile
    └── .env.example
```

## Reproducing the paper

`experiments/results.ipynb` globs every `*.csv` under `experiments/results/` into one DataFrame and rebuilds every table and chart in the [paper](https://arxiv.org/abs/2601.07978) end-to-end. Each chapter renders a `constrained − unconstrained` Δ panel underneath its bar chart so the regime impact per backend is always visible.

## Citation

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

LOCOMO benchmark dataset:

```bibtex
@article{maharana2024evaluating,
  title={Evaluating very long-term conversational memory of llm agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  journal={arXiv preprint arXiv:2402.17753},
  year={2024}
}
```

## License

See [LICENSE.txt](LICENSE.txt).
