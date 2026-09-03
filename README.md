# Cost and accuracy of long-term memory in distributed LLM-based multi-agent systems

This repository contains the testbed and the analysis used to produce the empirical results of the paper _Cost and Accuracy of Long-Term Memory in Distributed Multi-Agent Systems Based on Large Language Models_. The published paper can be found on [IEEE Xplore](https://ieeexplore.ieee.org/document/11645496). A preprint is available on [arXiv](https://arxiv.org/abs/2601.07978).

## Overview

When LLM agents need to remember things across long conversations, they rely on a _long-term memory_ system. Several such systems exist, each with a different idea of how memories should be stored and searched. This repository measures how three of them, plus two simple baselines, compare on answer accuracy and operating cost. The benchmark is [LoCoMo](https://github.com/snap-research/locomo), the setting is a realistic cloud-edge deployment.

| Name                                            | How it remembers                                      |
| ----------------------------------------------- | ----------------------------------------------------- |
| [cognee](https://github.com/topoteretes/cognee) | Graph plus vector embeddings, populated by an LLM     |
| [Graphiti](https://github.com/getzep/graphiti)  | A temporal knowledge graph                            |
| [Mem0](https://github.com/mem0ai/mem0)          | LLM-extracted facts in a vector store                 |
| RAG _(baseline)_                                | Raw conversation turns in a vector store, no LLM step |
| full-context _(baseline)_                       | The whole conversation, no compression at all         |

The paper asks whether the extra machinery of a memory framework actually buys better answers, and how that bet shifts when the link between the edge agent and the cloud becomes slow or constrained.

## Repository layout

```
dmas-memory/
├── dmas/                   # Coordinator, responder, memory, benchmark services
├── experiments/
│   ├── results/            # Per-experiment CSV outputs
│   └── analysis/
│       ├── results.ipynb       # Reproduces every table and figure of the paper
│       ├── requirements.txt    # Python dependencies for the notebook
│       └── figures/            # PDF figures emitted by the notebook
├── Makefile
└── .env.example
```

The benchmark is a Docker Compose stack organised into edge, cloud, and management networks; see [Network architecture](#network-architecture).

## Reproducing the analysis

`experiments/analysis/results.ipynb` ingests every CSV under `experiments/results/` and rebuilds, end-to-end, every table and figure reported in the paper: macro retention metrics, retrieval-failure rates, per-category accuracy, the system-level cost decomposition per phase, total cost of ownership, the Pareto frontier, and the per-framework network sensitivity tests.

```bash
cd experiments/analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook results.ipynb
```

LaTeX strings for the cost and retention tables are printed in place; PDF figures are written to `figures/`.

## Reproducing the experiments

This section covers end-to-end execution of the benchmark: building the images, bringing up the stack, and running the sweep that produces the CSVs consumed by the notebook above.

### Prerequisites

- Docker and Docker Compose
- GNU Make
- An OpenAI API key
- Approximately 12 GB of RAM

On Linux and macOS, `sudo bash install.sh` installs Make and Docker.

### Quick start

```bash
cp .env.example .env          # set OPENAI_API_KEY=sk-...
make build
make start

make experiment-test CONV=0   # smoke test on one conversation
make experiment               # full sweep
```

### Smoke targets

| Target                    | Messages | Questions    | Duration |
| ------------------------- | -------- | ------------ | -------- |
| `make experiment-test-s`  | 5        | 1 (cat 2)    | minutes  |
| `make experiment-test`    | 119      | 3 × cats 1-4 | ~hour    |
| `make experiment-test-l`  | 199      | 3 × cats 1-4 | hours    |

Each target sweeps both network scenarios for one `CONV` (default 0), reuses the load across scenarios (`KEEP_STATE=1`), and runs one judge call per answer. Restrict the set of implementations with `BACKENDS="mem0 graphiti"`.

### Make targets

| Command               | Effect                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `make build`          | Build every image from scratch. Aborts if `OPENAI_API_KEY` is missing.                       |
| `make start`          | Bring the full stack up. Langfuse public and secret keys are generated on first run.         |
| `make stop`           | Stop containers; volumes are preserved.                                                       |
| `make clean`          | Stop and drop only the memory volumes (`qdrant-data`, `neo4j-data`, `neo4j-logs`).           |
| `make reset`          | Stop and drop every named volume. Compose-managed bridges are torn down by `stop` and recreated on the next `start`. |
| `make experiment`     | Full sweep: 10 LoCoMo conversations × {unconstrained, constrained} × 5 implementations × 3-judge majority vote. |
| `make experiment-leg` | Single `(CONV, MODE)` × implementations. Knobs: `CONV MODE BACKENDS QUESTIONS MESSAGES Q_PER_TYPE QUESTION_TYPES KEEP_STATE`. |
| `make logs` / `ps`    | Tail logs / list containers.                                                                  |

### Running the full sweep across multiple machines

Each machine runs one slice of `BACKENDS`. `make experiment` with default arguments covers all 10 LoCoMo conversations across both scenarios. It auto-launches a tmux session named `exp`, traps SIGINT so that Ctrl-C inside the pane is a no-op, and tees output to `exp_<backends>.log`. Detach with `Ctrl-b d`; re-run the same command (or `tmux attach -t exp`) to re-attach.

```bash
make experiment BACKENDS=mem0                   # Machine 1
make experiment BACKENDS=graphiti               # Machine 2
make experiment BACKENDS=cognee                 # Machine 3
make experiment BACKENDS="rag full_context"    # Machine 4
```

CSVs are written to `experiments/results/{backend}_{mode}.csv`.

### Cleaning up

- `make stop`: kills the `exp` tmux session (if any) and brings the stack down. State is preserved.
- `make clean`: stop and drop only the memory backend volumes (qdrant, neo4j).
- `make reset`: stop and drop every named volume and remove top-level `*.log` files and every entry under `experiments/results/` except `backup/`. Move any results worth keeping into `experiments/results/backup/` before running this.

## Network architecture

Three Docker Compose bridges share a single gateway:

- `edge-net`: `coordinator`, `ollama`, `litellm-edge`
- `cloud-net`: `responder`, `memory`, `qdrant`, `neo4j`, `litellm-cloud`
- `mgmt-net`: `benchmark`, `langfuse-*` (observability and orchestration, never on the data plane)

`toxiproxy` is the only container with an interface in both data subnets. The coordinator is the only caller routed through it; `responder` ↔ `memory` and `memory` ↔ storage traffic stays direct on `cloud-net`.

`make build` selects a non-colliding /16 from a candidate list (172.30, 172.40, 10.42, 192.168.220, …) and pins the chosen subnets and toxiproxy IPs into `.env`. To pin manually, uncomment the corresponding block in `.env.example`.

## Network fault injection

The unconstrained scenario clears all toxics. The constrained scenario applies `CONSTRAINED_LATENCY`, `CONSTRAINED_JITTER`, and `CONSTRAINED_BANDWIDTH` (defaults 150 ms, 30 ms, 512 KB/s) to the toxiproxy memory and responder proxies, i.e. only to coordinator ↔ cloud traffic. The benchmark service re-verifies live toxic state mid-run and rejects with HTTP 412 on drift (`dmas/benchmark/app/toxics.py`).

## Network metering

Cross-boundary traffic is read at the toxiproxy `edge-net` veth: `rx_bytes` is edge → cloud (`network_edge_to_cloud_bytes`) and `tx_bytes` is cloud → edge (`network_cloud_to_edge_bytes`). A single chokepoint avoids double-counting and intra-cloud inflation. CPU, RAM, and disk are summed per `group=edge|cloud` label in `dmas/benchmark/app/cgroup_metrics.py`.

## Citation

A BibTeX entry will be provided once the IEEE COMPSAC 2026 publication is available.

LoCoMo benchmark dataset:

```bibtex
@inproceedings{maharana-etal-2024-evaluating,
    title     = "Evaluating Very Long-Term Conversational Memory of {LLM} Agents",
    author    = "Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei",
    editor    = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = aug,
    year      = "2024",
    address   = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.acl-long.747/",
    doi       = "10.18653/v1/2024.acl-long.747",
    pages     = "13851--13870"
}
```

## License

See [LICENSE.txt](LICENSE.txt).
