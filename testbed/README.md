# dmas-memory testbed

## Running the full experiment sweep across four machines

Each machine runs one slice of `BACKENDS`. `make experiment` with defaults
covers all 10 LOCOMO conversations × both modes (unconstrained + constrained).
Each command opens a tmux session named `exp`, streams logs live, and tees
them to a log file. Detach with `Ctrl-b d`; re-run the same command to
re-attach (`-A` attaches if the session already exists).

**Machine 1 (mem0)**
```bash
tmux new -A -s exp "make experiment BACKENDS=mem0 2>&1 | tee -a exp_mem0.log"
```

**Machine 2 (graphiti)**
```bash
tmux new -A -s exp "make experiment BACKENDS=graphiti 2>&1 | tee -a exp_graphiti.log"
```

**Machine 3 (cognee)**
```bash
tmux new -A -s exp "make experiment BACKENDS=cognee 2>&1 | tee -a exp_cognee.log"
```

**Machine 4 (rag + full_context)**
```bash
tmux new -A -s exp "make experiment BACKENDS='rag full_context' 2>&1 | tee -a exp_rag_fullctx.log"
```

Results land in `experiments/results/{backend}_{mode}.csv`.

## Cleaning up

- `make stop` — bring the stack down, keep all state.
- `make clean` — stop + drop the memory backend volumes (qdrant, neo4j).
- `make reset` — stop + drop every named volume + remove top-level `*.log`
  and every entry under `experiments/results/` except `backup/`. Move any
  results worth keeping into `experiments/results/backup/` before running.
