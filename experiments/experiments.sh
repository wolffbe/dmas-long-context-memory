#!/usr/bin/env bash
# Thin wrapper around `make experiment` — the publishable full sweep
# now lives in the Makefile so all knobs (CONVS, BACKENDS,
# LLM_AS_JUDGE_SEED) compose the same way whether you invoke it from
# make or this script.
#
# Knobs (passed straight to make):
#   CONVS="0 1 2 3 4 5 6 7 8 9"  bash experiments.sh
#   BACKENDS="mem0 graphiti"     bash experiments.sh
#   LLM_AS_JUDGE_SEED=5          bash experiments.sh

set -eu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

exec make experiment "$@"
