#!/usr/bin/env bash
# Regenerate ABLATION-001 derived analyses from the immutable raw/ artifacts,
# using the committed producing code. READ-ONLY with respect to raw/ and to
# MAIN-001. Does not re-run the experiment (that is deterministic given the
# frozen protocol; see config.json.reproducibility).
#
# Prereqs: check out git 4533c61 (tag ablation-001-frozen) in ../../code, and
# use the rova environment (Python 3.13.14, numpy 2.5.1, scipy 1.18.0).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CODE="$HERE/../../code"
RAW="$HERE/raw"
OUTDIR="${1:-$HERE/analysis_reproduced}"   # never overwrites the archived analysis/

cd "$CODE"
PY="${PYTHON:-python3}"

echo "[1/3] consistency validator (hard gate; vs archived MAIN-001 ROVA, read-only)"
"$PY" -m dtcoa_val.experiments.validate_consistency \
  --ablation-out "$RAW" --report-out "$OUTDIR/consistency_report.json"

echo "[2/3] DR3 descriptive (recompute from raw + cross-check driver reports)"
"$PY" -m dtcoa_val.analysis.dr3_attribution \
  --ablation-out "$RAW" --out-dir "$OUTDIR/dr3_descriptive"

echo "[3/3] DR3 regime inference (Option B + C)"
"$PY" -m dtcoa_val.analysis.dr3_regime_inference \
  --ablation-out "$RAW" --out-dir "$OUTDIR/dr3_regime"

echo "Done. Compare $OUTDIR against the archived analysis/ (should match modulo timestamps)."
