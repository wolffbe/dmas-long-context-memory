#!/usr/bin/env bash
# Run both modes (unconstrained, constrained) for one CONV across all
# backends. The benchmark wipes memory state per backend so the legs are
# fully independent — no manual reset between modes.
#
# Knobs:
#   CONV=0  SEEDS=1  bash experiments.sh
#   BACKENDS="mem0 graphiti" bash experiments.sh

set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONV="${CONV:-0}"
SEEDS="${SEEDS:-1}"
BACKENDS="${BACKENDS:-mem0 graphiti rag full_context}"

for mode in unconstrained constrained; do
  echo "==> $mode leg (CONV=$CONV BACKENDS='$BACKENDS' SEEDS=$SEEDS)"
  make experiment CONV="$CONV" MODE="$mode" SEEDS="$SEEDS" BACKENDS="$BACKENDS"
done

echo "==> done. rows in experiments/results/results.csv"
