"""mcns 3b: the male CNS brain drives the walking fly in the live engine (closed loop over fly_link's host transport).

Per control step (K = 10 physics steps, 1 ms): the resident MCNS Shiu brain advances K ticks; the 381 leg motor neurons'
spikes of the last 100 ms, per side, go through the ridge readout fitted by tools/mcns_mn_drive.py (.work/agent/mcns/
mn_readout.npz, z-scored counts -> commanded drive per side) into fly_link's gait mapping (FlyGym rhythm, 50 ctrl).
The brain is commanded as in the fit: forward DNs 80 Hz; a turn = one side's forward DNs +100 Hz (the VNC has no
rhythm and DNa01/02 do not lateralize the MNs, tools/mcns_dna_reach.py, so the turn command is a stand-in, not DNa).
Columns: sym, left, right; engines: sym, left, right, silenced (MN counts zeroed), mirrored (readout sides swapped: mutant).
Gates (1 s): sym walks > 1 body length, tilt < 20 deg; left and right turn opposite ways, each beyond 10 deg; silenced
moves < 25 % of sym; mirrored: left command turns the other way (sign flips vs 'left').
  gpu-wait run --shared --vram 4 --priority 60 --timeout 3000 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.mcns_link
"""
import json
import sys

import numpy as np

from agent.body_engine.fly_link import BODY_LENGTH, DT, K, OUT, STEPS, Engine, Gait, metrics, prepare

WINDOW, READOUT = 100, 'mcns/mn_readout.npz'
FWD_HZ, TURN_HZ = 80, 100  # drive 80/140 = 0.57 forward; a turn adds 0.5 on its side (1.0 vs 0.57): at 140 Hz both clip to 1 and the turn vanishes


def brain():
    from agent import mcns, shiu_g1
    from agent.mcns_mapping import descending_name_sets
    from agent.shiu import ShiuLive
    data = mcns.graph()
    n = data[4][0]
    dn, _ = descending_name_sets()
    b = ShiuLive(*next(shiu_g1.arms(data, 'mcns'))[1], n=n, flies=3, seed=7)
    for col, t in enumerate((None, 'left', 'right')):
        b.stimulate(dn['forward'], FWD_HZ, fly=col, key=col)
        if t:
            b.stimulate(dn['forward_' + t], FWD_HZ + TURN_HZ, fly=col, key=col)
    f = np.load(f'.work/agent/{READOUT}')
    cells = np.r_[f['mnL'], f['mnR']]
    b.start([1], np.concatenate([c * n + cells for c in range(3)]).astype(np.int32))
    return b, f, len(f['mnL']), len(cells)


def drive(f, x, nl):
    """per-cell MN counts of one 100 ms window -> (left, right) drive in [0, 1]."""
    return np.clip([((x[:nl] - f['mu0']) / f['sd0']) @ f['w0'] + f['y0'], ((x[nl:] - f['mu1']) / f['sd1']) @ f['w1'] + f['y1']], 0, 1)


def main():
    d, m, _ = prepare()
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    names = ('sym', 'left', 'right', 'silenced', 'mirrored')
    gait = {k: Gait(m, d, K) for k in names}
    eng = {k: Engine(d, K, 'mcns-' + k, STEPS) for k in names}
    b, f, nl, nc = brain()
    cum, tr = [np.zeros(3 * nc)], {k: [q0] for k in names}
    for i in range(STEPS):
        cum.append(b.advance(K).astype(float))
        x = (cum[-1] - cum[-1 - min(i + 1, WINDOW)]).reshape(3, nc) * (WINDOW / min(i + 1, WINDOW))  # counts per 100 ms
        dr = [drive(f, x[c], nl) for c in range(3)] if i + 1 >= WINDOW else [np.zeros(2)] * 3  # warm-up: stand
        sig = {'sym': dr[0], 'left': dr[1], 'right': dr[2], 'silenced': np.zeros(2), 'mirrored': dr[1][::-1]}
        for k in names:
            eng[k].send(gait[k].ctrl(sig[k]))
        for k in names:
            tr[k].append(eng[k].recv())
    b.close()
    for k in names:
        eng[k].close()
    r = {k: metrics(m, None, np.array(tr[k])) for k in names}
    fw = {k: float(r[k]['forward'][-1]) for k in names}
    yaw = {k: float(r[k]['yaw_deg'][-1]) for k in names}
    print(json.dumps({'forward_mm': fw, 'yaw_deg': yaw, 'tilt_max_sym': r['sym']['tilt_max']}))
    ok = [(f"walk: sym {fw['sym']:.2f} mm > {BODY_LENGTH} and tilt {r['sym']['tilt_max']:.1f} < 20", fw['sym'] > BODY_LENGTH and r['sym']['tilt_max'] < 20),
          (f"turn: left {yaw['left']:+.1f}, right {yaw['right']:+.1f} deg, opposite, each beyond 10", yaw['left'] * yaw['right'] < 0 and min(abs(yaw['left']), abs(yaw['right'])) > 10),
          (f"silenced: {fw['silenced']:.2f} mm < 25 % of sym", abs(fw['silenced']) < 0.25 * fw['sym']),
          (f"mutant mirrored readout: yaw {yaw['mirrored']:+.1f} vs left {yaw['left']:+.1f}, sign flips", yaw['mirrored'] * yaw['left'] < 0)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    (OUT / 'mcns-link.json').write_text(json.dumps({'forward_mm': fw, 'yaw_deg': yaw}))
    if not all(v for _, v in ok):
        sys.exit('FAIL mcns closed loop')
    print('PASS mcns closed loop: the male CNS leg-MN readout walks and turns the fly')


if __name__ == '__main__':
    main()
