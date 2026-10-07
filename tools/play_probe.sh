#!/usr/bin/env bash
# Forced one-sided loom probe paired over seeds (docs/plans/PLAY.md). usage: tools/play_probe.sh [N_SEEDS=3] [swap=0|1]
# PASS: every seed's (see - blind) yaw delta has the right sign (left loom -> right turn: negative; right loom: positive) and the
# median over seeds is beyond 10 deg on both sides. swap=1 must FAIL. One gpu-wait job per seed.
set -euo pipefail
n=${1:-3}; swap=${2:-0}; out=.work/review/probe-$swap; mkdir -p "$out"
for s in $(seq 1 "$n"); do
  gpu-wait run --priority 60 --vram 4 --timeout 900 --env PLAY_LOOM_FORCE=1 --env PLAY_LOOM_SWAP=$swap -- .venv/bin/python -m agent.body_engine.play walls "$s" > "$out/seed$s.log" 2>&1 || true
  grep -q yaw_deg "$out/seed$s.log" || { echo "FAIL probe seed $s: no receipt, $out/seed$s.log"; exit 1; }
done
.venv/bin/python - "$out" "$n" <<'PY'
import json, re, sys, numpy as np
out, n = sys.argv[1], int(sys.argv[2])
d = []
for s in range(1, n + 1):
    y = json.loads(re.search(r'\{"yaw_deg.*', open(f'{out}/seed{s}.log').read()).group(0))['yaw_deg [see L, see R, blind L, blind R]']
    d.append((y[0] - y[2], y[1] - y[3])); print(f"seed {s}: see-blind L {d[-1][0]:+.1f} R {d[-1][1]:+.1f}")
d = np.array(d); med = np.median(d, 0)
ok = (d[:, 0] < 0).all() and (d[:, 1] > 0).all() and med[0] < -10 and med[1] > 10
print(('PASS' if ok else 'FAIL'), f"probe paired over {n} seeds: median L {med[0]:+.1f} R {med[1]:+.1f} (signs all right, medians beyond 10)")
sys.exit(0 if ok else 1)
PY
