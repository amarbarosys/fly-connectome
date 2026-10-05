"""mcns unit 3a (user decision 2026-10-05, option a): the gait mapping keeps the rhythm (the MCNS Shiu VNC has none,
tools/mcns_mn_rhythm.py); the leg motor neurons, read per side, replace the descending-neuron readout as the two drives
the gait mapping takes. This step learns that readout and checks it on held-out conditions.

Conditions (one column each, one launch, 2 s): forward DNs at 50-200 Hz (6 levels), symmetric or a turn = one side's
forward DNs at +50 or +100 Hz (30 conditions; the 12-condition first version gave R^2 0.705 / 0.830 on two runs).
Per 100 ms window after a 200 ms onset: teacher = the commanded drive per side (forward rate / 150, plus 0.5 on the
turning side); input = the leg MN counts per side. (The DN spike readout as teacher was tried first: the 4 turning DNs
vanish in the 784 forward DNs' sum, asymmetry +-0.04, held-out R^2 -81.)
Fit: ridge per side (left MNs -> left drive, right MNs -> right drive), leave-one-condition-out.
Gates: held-out R^2 of the asymmetry (left - right drive) >= 0.7 and turn sign right in every turning condition;
mutant: the same fit against a condition-permuted teacher -> asymmetry R^2 < 0.3; reported: unlearned per-side sums.
usage: gpu-wait run --shared --vram 3 --priority 60 --timeout 900 --env PYTHONPATH=$PWD -- .venv/bin/python tools/mcns_mn_drive.py
"""
import json
import sys

import numpy as np

from agent import mcns, shiu_g1
from agent.mcns_mapping import descending_name_sets, leg_motor_groups
from agent.shiu import ShiuGPU

FWD, TURN = (50, 80, 110, 140, 170, 200), (None, 'left', 'right')
DT = (50, 100)  # turn sizes: that side's forward DNs + DT Hz
COND = [(f, None, 0) for f in FWD] + [(f, t, d) for f in FWD for t in TURN[1:] for d in DT]  # 30 conditions
data = mcns.graph()
graph = next(shiu_g1.arms(data, 'mcns'))[1]
n = data[4][0]
dn, _ = descending_name_sets()
mg, _ = leg_motor_groups()
mn = {s: np.concatenate([mg[k] for k in mg if k.endswith('_' + s)]) for s in 'LR'}
dns = {'L': np.r_[dn['forward_left'], dn['turn_left']], 'R': np.r_[dn['forward_right'], dn['turn_right']]}
brain = ShiuGPU(*graph, n=n, flies=len(COND), seed=7)
for c, (f, t, d) in enumerate(COND):
    brain.stimulate(dn['forward'], f, fly=c)
    if t:  # turn = one side's forward DNs 100 Hz harder (DNa01/02 at 150 Hz moved MN sides by noise only: 1911/1611 L vs 1917/1574 R)
        brain.stimulate(dn['forward_' + t], f + d, fly=c)
cells = np.r_[mn['L'], mn['R'], dns['L'], dns['R']]
nl, nr, dl = len(mn['L']), len(mn['R']), len(dns['L'])
readout = np.concatenate([c * n + cells for c in range(len(COND))]).astype(np.int32)
win = np.arange(2000, 20001, 1000)  # 200 ms onset, then 18 windows of 100 ms (540 samples for 381 MNs)
cg = np.full(n, -1, np.int32); cg[cells] = 0
_, got = brain.run_groups(win, cg, readout)
x = np.diff(got.astype(np.int64), axis=0).reshape(len(win) - 1, len(COND), len(cells))  # (8 windows, conditions, cells)
mnL, mnR = x[..., :nl], x[..., nl:nl + nr]
dnL, dnR = x[..., nl + nr:nl + nr + dl].sum(-1), x[..., nl + nr + dl:].sum(-1)
ref = COND.index((140, None, 0))
cmd = np.array([[f / 140 + d / 200 * (t == 'left'), f / 140 + d / 200 * (t == 'right')] for f, t, d in COND])
drive = np.broadcast_to(cmd, (len(win) - 1, *cmd.shape)).astype(float)  # (w, c, 2) teacher
side = {f'{f}Hz {t} {d}': [int(mnL[:, i].sum()), int(mnR[:, i].sum())] for i, (f, t, d) in enumerate(COND)}  # MN spikes L, R


def fit_eval(xl, xr, lam=10.0):
    """leave-one-condition-out ridge per side; returns held-out predicted drives (w, c, 2)."""
    pred = np.zeros_like(drive)
    for c in range(len(COND)):
        tr = [k for k in range(len(COND)) if k != c]
        for s, X in enumerate((xl, xr)):
            A = X[:, tr].reshape(-1, X.shape[-1]).astype(float); y = drive[:, tr, s].ravel()
            mu, sd = A.mean(0), A.std(0) + 1e-9
            Z = (A - mu) / sd
            wgt = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
            pred[:, c, s] = ((X[:, c] - mu) / sd) @ wgt + y.mean()
    return pred


def r2(p):
    a, b = drive[..., 0] - drive[..., 1], p[..., 0] - p[..., 1]
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


p = fit_eval(mnL, mnR)
signs = {f'{f}Hz {t} +{d}': bool(np.sign((p[:, i, 0] - p[:, i, 1]).mean()) == (1 if t == 'left' else -1))
         for i, (f, t, d) in enumerate(COND) if t}
allmn = np.concatenate([mnL, mnR], -1)
# mutant: the teacher's conditions permuted (a random-side MN split is no control: signed ridge weights on a mixed half
# still read the asymmetry, R^2 0.81), so a fit that only overfits 190 MNs on 96 samples fails it
real, drive = drive, drive[:, np.random.default_rng(3).permutation(len(COND))]
q = fit_eval(mnL, mnR)
drive = real
sums = np.stack([mnL.sum(-1) / mnL[:, ref].sum(-1).mean(), mnR.sum(-1) / mnR[:, ref].sum(-1).mean()], -1)
a, b = drive[..., 0] - drive[..., 1], sums[..., 0] - sums[..., 1]
corr_sum = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])  # unlearned anatomical side sums vs commanded asymmetry
out = {'mn_spikes_LR': side, 'active_mn': int((allmn.sum((0, 1)) > 0).sum()), 'mn': nl + nr,
       'asym_r2': r2(p), 'turn_sign': signs, 'mutant_permuted_teacher_asym_r2': r2(q), 'unlearned_side_sum_corr': corr_sum,
       'teacher_asym_range': [float((drive[..., 0] - drive[..., 1]).min()), float((drive[..., 0] - drive[..., 1]).max())]}
print(json.dumps(out))
ok = [('G1 held-out asymmetry R^2 >= 0.7', out['asym_r2'] >= 0.7), ('G2 turn sign in every turning condition', all(signs.values())),
      ('M1 permuted-teacher fit asymmetry R^2 < 0.3', out['mutant_permuted_teacher_asym_r2'] < 0.3)]
for k, v in ok:
    print(('PASS' if v else 'FAIL'), k)
if not all(v for _, v in ok):
    sys.exit('FAIL mcns MN drive')
print('PASS mcns MN drive readout')
