#!/usr/bin/env bash
# REV-002 A1 runner (docs/REVISION_002_ANALYSIS_PLAN.md, section 2). Usage:
#   bash rova_revision/run_a1.sh check  OUT_ROOT REV001_DIR   tests + A1 environment gate + REV-001 input check
#   bash rova_revision/run_a1.sh full   OUT_ROOT REV001_DIR   scientific run -> OUT_ROOT/a1_out
#   bash rova_revision/run_a1.sh verify OUT_ROOT              verify OUT_ROOT/a1_out
# REV001_DIR is the REV-001 revision_out directory (read only; never written).
# Optional environment: ROVA_WORKERS (default 15), ROVA_ENV_APPROVAL (recorded verbatim;
# use only with a written approval for this run).
set -euo pipefail

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:?mode: check | full | verify}"
OUT_ROOT="${2:?output root (outside the repository)}"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
WORKERS="${ROVA_WORKERS:-15}"
APPROVAL_ARGS=()
if [ -n "${ROVA_ENV_APPROVAL:-}" ]; then APPROVAL_ARGS=(--approval "$ROVA_ENV_APPROVAL"); fi

case "$MODE" in
  check)
    REV001="${3:?REV-001 revision_out directory}"
    python -m rova_revision.tests.test_core_modules
    python -m rova_revision.tests.test_matched_worlds
    python -m rova_revision.tests.test_env_gate
    python -m rova_revision.tests.test_decision_rules
    python -m rova_revision.tests.test_orchestration_analysis
    python -m rova_revision.tests.test_scientific_paths
    python -m rova_revision.tests.test_rev002
    git status --porcelain
    python -m rova_revision.a1_guard --workers "$WORKERS" --json "$OUT_ROOT/a1_env_guard_check.json" \
      ${ROVA_ENV_APPROVAL:+--approval "$ROVA_ENV_APPROVAL"} | tee "$OUT_ROOT/a1_env_guard_check.txt" \
      && echo "A1 environment gate: PASSED ($OUT_ROOT/a1_env_guard_check.txt)" \
      || { echo "A1 environment gate: FAILED ($OUT_ROOT/a1_env_guard_check.txt)"; exit 2; }
    python -c "import json,sys; from rova_revision.clustered import verify_rev001_dir; r=verify_rev001_dir(sys.argv[1]); print(json.dumps({k: r[k] for k in ('sha256sums_sha256','expected','entries','expected_entries','passed')}, indent=1)); sys.exit(0 if r['passed'] else 4)" "$REV001" \
      && echo "REV-001 inputs: VERIFIED" || { echo "REV-001 inputs: FAILED"; exit 4; }
    ;;
  full)
    REV001="${3:?REV-001 revision_out directory}"
    python -m rova_revision.run_a1 --profile full --out "$OUT_ROOT/a1_out" --rev001 "$REV001" \
      --workers "$WORKERS" "${APPROVAL_ARGS[@]}"
    ;;
  verify)
    python -m rova_revision.verify_a1 --profile full --out "$OUT_ROOT/a1_out"
    ;;
  *)
    echo "unknown mode: $MODE" >&2; exit 64 ;;
esac
