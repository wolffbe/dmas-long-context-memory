# dmas-memory testbed

## Running the full experiment sweep across four machines

Each machine runs one slice of `BACKENDS`. `make experiment` with defaults
covers all 10 LOCOMO conversations × both modes (unconstrained + constrained).
It auto-launches a tmux session named `exp`, traps SIGINT so Ctrl+C inside
the pane is a no-op, and tees output to `exp_<backends>.log`. Detach with
`Ctrl-b d`; re-run the same command (or `tmux attach -t exp`) to re-attach.

**Machine 1 (mem0)**
```bash
make experiment BACKENDS=mem0
```

**Machine 2 (graphiti)**
```bash
make experiment BACKENDS=graphiti
```

**Machine 3 (cognee)**
```bash
make experiment BACKENDS=cognee
```

**Machine 4 (rag + full_context)**
```bash
make experiment BACKENDS="rag full_context"
```

Results land in `experiments/results/{backend}_{mode}.csv`.

## Cleaning up

- `make stop` — kill the `exp` tmux session (if any) and bring the stack
  down. State is preserved.
- `make clean` — stop + drop the memory backend volumes (qdrant, neo4j).
- `make reset` — stop + drop every named volume + remove top-level `*.log`
  and every entry under `experiments/results/` except `backup/`. Move any
  results worth keeping into `experiments/results/backup/` before running.
