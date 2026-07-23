#!/usr/bin/env bash
# reproduce.sh — one-shot driver for the ROVA_CCE monotonicity reproduction
# package. Runs both replay scripts against the ARCHIVED code and experiments,
# tees console output to logs, and prints sha256 of every generated artifact.
#
# Usage:   bash reproduce/reproduce.sh
# Env:     Python 3 with numpy (see README for the archived environment).
#
# No source files are modified. All outputs land in this reproduce/ directory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
CODE="$REPO/code"
export PYTHONPATH="$CODE"
# Pin BLAS threads so numpy reductions are deterministic across machines.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=================================================================="
echo "ENVIRONMENT"
echo "=================================================================="
python3 --version
python3 -c "import numpy, sys; print('numpy', numpy.__version__)"
echo "repo : $REPO"
echo "code : $CODE"
echo

echo "=================================================================="
echo "[1/2] replay_monotonicity.py  (minimal synthetic counterexamples)"
echo "=================================================================="
python3 "$HERE/replay_monotonicity.py" 2>&1 | tee "$HERE/_monotonicity.log"
echo

echo "=================================================================="
echo "[2/2] replay_archived_runs.py (MAIN-001 + ABLATION-001 replay)"
echo "=================================================================="
python3 "$HERE/replay_archived_runs.py" --experiment both --print-events 8 \
    2>&1 | tee "$HERE/_archived.log"
echo

echo "=================================================================="
echo "ARTIFACT HASHES (sha256)"
echo "=================================================================="
for f in out_monotonicity.json out_archived.json _monotonicity.log _archived.log; do
    if [ -f "$HERE/$f" ]; then
        sha256sum "$HERE/$f"
    fi
done
echo
echo "Done. Compare the printed counts against reproduce/expected_outputs.md."
