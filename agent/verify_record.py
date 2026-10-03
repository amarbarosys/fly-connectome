"""Full-brain recording (FLY_RECORD) must hold the same counts the worker's own readout returns: for 3 sniffs x 2
windows x 4 flies, every readout cell's recorded count equals gather.bin, and the recording covers all n neurons.

usage: gpu-wait run --timeout 600 -- .venv/bin/python -m agent.verify_record
"""
import numpy as np
from agent.learning_loop import setup
from agent.verify_pain import brain
from agent.shiu import ShiuLive, ROOT
pre, post, w, n, grp, kc, approach, avoid, odour = setup()
b = brain(ShiuLive, pre, post, w, n, grp, kc, odour, 4)
path = ROOT / '.work/agent/shiu/record-check.bin'
path.unlink(missing_ok=True)
readout = np.concatenate([np.asarray(kc[:50]) + c * b.n for c in range(4)])  # column * n + neuron
b.start([100, 200], readout, record=path)
_, p = b._stim_arrays()
got = [b.sniff(p) for _ in range(3)]
b.close()
rec = b.recording()  # (sniffs, windows, n, flies)
flat = rec.transpose(0, 1, 3, 2).reshape(3, 2, -1)  # column * n + neuron
bad = sum(int((flat[s][:, readout] != got[s]).sum()) for s in range(3))
active = int((rec[-1, -1] > 0).sum())
ok = rec.shape == (3, 2, b.n, 4) and bad == 0 and active > 0
print(f'recording {rec.shape}, {path.stat().st_size / 1e6:.1f} MB; readout mismatches {bad}; '
      f'{active} neurons x flies spiked in the last window; {"PASS" if ok else "FAIL"}')
raise SystemExit(0 if ok else 1)
