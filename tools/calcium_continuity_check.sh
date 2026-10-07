#!/usr/bin/env bash
# Calcium replay continuity gate and its falsifier: a reset-per-window mutant must fail the gate.
# CPU only, about 10 s. usage: tools/calcium_continuity_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
gate=(env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .work/fly-research/arena-venv/bin/python -m agent.verify_calcium_validation --continuity)
"${gate[@]}" || { echo "FAIL continuity: gate red on the real harness"; exit 1; }
MUTANT_EXPECT=state_l1 ~/iTools/dev/mutant-check/mutant-check.sh agent/calcium_validation.py \
  "s|^\(                current_trial = int(protocol\['trial'\]\[i\])\)\$|\1; brain.reset()|" -- "${gate[@]}" \
  || { echo "FAIL continuity: reset-per-window mutant passed the gate"; exit 1; }
echo "PASS continuity: windows start from carried state; reset-per-window mutant caught"
