# Testbed setup and execution

This document covers building, starting, and operating the benchmark stack. Repository-level information (what the project is and how to reproduce the analysis) lives in the [top-level README](README.md).

## Prerequisites

- Docker and Docker Compose
- GNU Make
- An OpenAI API key
- Approximately 12 GB of RAM

On Linux and macOS, `sudo bash install.sh` installs Make and Docker.

## Quick start

```bash
cp .env.example .env          # set OPENAI_API_KEY=sk-...
make build
make start

make experiment-test CONV=0   # smoke test on one conversation
make experiment               # full sweep
```

## Smoke targets

| Target                    | Messages | Questions    | Duration |
| ------------------------- | -------- | ------------ | -------- |
| `make experiment-test-s`  | 5        | 1 (cat 2)    | minutes  |
| `make experiment-test`    | 119      | 3 × cats 1-4 | ~hour    |
| `make experiment-test-l`  | 199      | 3 × cats 1-4 | hours    |

Each target sweeps both network scenarios for one `CONV` (default 0), reuses the load across scenarios (`KEEP_STATE=1`), and runs one judge call per answer. Restrict the set of implementations with `BACKENDS="mem0 graphiti"`.

## Make targets

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

## Running the full sweep across multiple machines

Each machine runs one slice of `BACKENDS`. `make experiment` with default arguments covers all 10 LoCoMo conversations across both scenarios. It auto-launches a tmux session named `exp`, traps SIGINT so that Ctrl-C inside the pane is a no-op, and tees output to `exp_<backends>.log`. Detach with `Ctrl-b d`; re-run the same command (or `tmux attach -t exp`) to re-attach.

```bash
make experiment BACKENDS=mem0                   # Machine 1
make experiment BACKENDS=graphiti               # Machine 2
make experiment BACKENDS=cognee                 # Machine 3
make experiment BACKENDS="rag full_context"    # Machine 4
```

CSVs are written to `experiments/results/{backend}_{mode}.csv`.

## Cleaning up

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
