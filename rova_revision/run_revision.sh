#!/usr/bin/env bash
# REV-001 runner. Usage:
#   bash rova_revision/run_revision.sh check OUT_ROOT     tests + environment guard
#   bash rova_revision/run_revision.sh pilot OUT_ROOT     timing pilot -> OUT_ROOT/pilot_out
#   bash rova_revision/run_revision.sh full  OUT_ROOT     scientific run -> OUT_ROOT/revision_out
#   bash rova_revision/run_revision.sh verify OUT_ROOT    checksums, unit digests, completeness, gate
# Optional environment: ROVA_TIERS (default A,B,C), ROVA_WORKERS (default 15),
# ROVA_ENV_APPROVAL (written approval of a soft environment mismatch; recorded verbatim).
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:?mode: check | pilot | full | verify}"
OUT_ROOT="${2:?output root (outside the repository)}"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
TIERS="${ROVA_TIERS:-A,B,C}"
WORKERS="${ROVA_WORKERS:-15}"
APPROVAL_ARGS=()
if [ -n "${ROVA_ENV_APPROVAL:-}" ]; then APPROVAL_ARGS=(--approval "$ROVA_ENV_APPROVAL"); fi

case "$MODE" in
  check)
    python -m rova_revision.tests.test_core_modules
    python -m rova_revision.tests.test_matched_worlds
    python -m rova_revision.tests.test_env_gate
    python -m rova_revision.tests.test_decision_rules
    python -m rova_revision.tests.test_orchestration_analysis
    python -m rova_revision.tests.test_scientific_paths
    git status --porcelain
    python -m rova_revision.env_guard --workers "$WORKERS" --json "$OUT_ROOT/env_guard_check.json" \
      ${ROVA_ENV_APPROVAL:+--approval "$ROVA_ENV_APPROVAL"} | tee "$OUT_ROOT/env_guard_check.txt" \
      && echo "environment gate: PASSED ($OUT_ROOT/env_guard_check.txt)" \
      || { echo "environment gate: FAILED ($OUT_ROOT/env_guard_check.txt)"; exit 2; }
    ;;
  pilot)
    python -m rova_revision.run_revision --profile pilot --out "$OUT_ROOT/pilot_out" "${APPROVAL_ARGS[@]}"
    python -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in ('estimated_core_hours_by_tier','estimated_total_core_hours','estimated_wall_hours_15_workers')}, indent=1))" "$OUT_ROOT/pilot_out/pilot_timing.json"
    ;;
  full)
    python -m rova_revision.run_revision --profile full --out "$OUT_ROOT/revision_out" \
      --workers "$WORKERS" --tiers "$TIERS" "${APPROVAL_ARGS[@]}"
    ;;
  verify)
    python -m rova_revision.verify_outputs --profile full --out "$OUT_ROOT/revision_out"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 64 ;;
esac
