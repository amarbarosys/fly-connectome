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


VIS = ('blank', 'left', 'right', 'mirrored')  # vision mode: forward 80 Hz + an image; disc at az +-70 (left/right eye), mirrored = left disc through the mirrored eye map (mutant)


def brain(vision=False, cols=None):
    from agent import mcns, shiu_g1
    from agent.mcns_mapping import descending_name_sets
    from agent.shiu import ShiuLive
    data = mcns.graph()
    n = data[4][0]
    dn, _ = descending_name_sets()
    late = cols is not None  # caller adds its own stimuli, then calls start
    cols = cols or (4 if vision else 3)
    b = ShiuLive(*next(shiu_g1.arms(data, 'mcns'))[1], n=n, flies=cols, seed=7)
    if vision:
        from agent import fly_vision as fv
        ids, az, el = fv.receptors(fv.LAMINA)
        img = {'blank': fv.np.full((90, 180), fv.BG), 'left': fv.disc(70), 'right': fv.disc(-70), 'mirrored': fv.disc(70)}
    for col, t in enumerate(VIS if vision else (None,) * cols if late else (None, 'left', 'right')):
        b.stimulate(dn['forward'], FWD_HZ, fly=col, key=col)
        if vision:
            for cells, hz in fv.drive(img[t], ids, -az if t == 'mirrored' else az, el):
                b.stimulate(cells, hz, fly=col, key=col)
        elif t:
            b.stimulate(dn['forward_' + t], FWD_HZ + TURN_HZ, fly=col, key=col)
    f = np.load(f'.work/agent/{READOUT}')
    cells = np.r_[f['mnL'], f['mnR']]
    if late:
        return b, f, len(f['mnL']), len(cells)
    b.start([1], np.concatenate([c * n + cells for c in range(cols)]).astype(np.int32))
    return b, f, len(f['mnL']), len(cells)


def drive(f, x, nl):
    """per-cell MN counts of one 100 ms window -> (left, right) drive in [0, 1]."""
    return np.clip([((x[:nl] - f['mu0']) / f['sd0']) @ f['w0'] + f['y0'], ((x[nl:] - f['mu1']) / f['sd1']) @ f['w1'] + f['y1']], 0, 1)


def vision():
    """fly vision in the walking body: a dark disc left or right of the fly (fixed on the retina, open loop) -> L1-L3/R1-6 ->
    MCNS brain -> leg-MN readout -> gait. Gates (1 s): left and right discs turn opposite ways, each >= 10 deg more than blank;
    blank walks > 1 body length; mutant mirrored eye map: the left disc turns like the right one (sign vs blank flips)."""
    d, m, _ = prepare()
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    eng, gait = {k: Engine(d, K, 'mcnsvis-' + k, STEPS) for k in VIS}, {k: Gait(m, d, K) for k in VIS}
    b, f, nl, nc = brain(vision=True)
    cum, tr = [np.zeros(4 * nc)], {k: [q0] for k in VIS}
    for i in range(STEPS):
        cum.append(b.advance(K).astype(float))
        x = (cum[-1] - cum[-1 - min(i + 1, WINDOW)]).reshape(4, nc) * (WINDOW / min(i + 1, WINDOW))
        for c, k in enumerate(VIS):
            eng[k].send(gait[k].ctrl(drive(f, x[c], nl) if i + 1 >= WINDOW else np.zeros(2)))
        for k in VIS:
            tr[k].append(eng[k].recv())
    b.close()
    for k in VIS:
        eng[k].close()
    r = {k: metrics(m, None, np.array(tr[k])) for k in VIS}
    fw = {k: float(r[k]['forward'][-1]) for k in VIS}
    yaw = {k: float(r[k]['yaw_deg'][-1]) for k in VIS}
    dl, dr, dm = yaw['left'] - yaw['blank'], yaw['right'] - yaw['blank'], yaw['mirrored'] - yaw['blank']
    print(json.dumps({'forward_mm': fw, 'yaw_deg': yaw}))
    ok = [(f"blank walks {fw['blank']:.2f} mm > {BODY_LENGTH}", fw['blank'] > BODY_LENGTH),
          (f"left disc {dl:+.1f}, right disc {dr:+.1f} deg vs blank: opposite, each >= 10", dl * dr < 0 and min(abs(dl), abs(dr)) >= 10),
          (f"mutant mirrored map: {dm:+.1f} vs left {dl:+.1f}, sign flips", dm * dl < 0)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    (OUT / 'mcns-vision.json').write_text(json.dumps({'forward_mm': fw, 'yaw_deg': yaw}))
    if not all(v for _, v in ok):
        sys.exit('FAIL fly vision in the walking body')
    print('PASS fly vision in the walking body: a disc on one side turns the fly')


SENSE = ('nosense', 'sense', 'shuffled')  # leg sense off; on; on with leg labels permuted across cells (mutant)


def sense():
    """leg touch and proprioception in the closed loop (agent/fly_legsense.py): each 1 ms the engine's qpos -> leg sensory
    rates -> MCNS brain (forward DNs 80 Hz) -> leg-MN readout -> gait. Per-leg specificity S = mean over legs of
    corr(leg touch, own-leg MN counts) - mean corr(leg touch, other legs' MNs), 10 ms bins, after the warm-up.
    Gates (pre-registered): sense walks > 1 body length, tilt < 20; S(sense) - S(nosense) >= 0.05; mutant: S(sense) - S(shuffled) >= 0.05."""
    from agent import fly_legsense as ls
    from agent.mcns_mapping import annotation_table
    d, m, _ = prepare()
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    eng, gait = {k: Engine(d, K, 'mcnssense-' + k, STEPS) for k in SENSE}, {k: Gait(m, d, K) for k in SENSE}
    feel = {k: ls.Sense(m, q0) for k in SENSE}
    groups = {'nosense': ls.cells(), 'sense': ls.cells(), 'shuffled': ls.cells(np.random.default_rng(5))}
    b, f, nl, nc = brain(cols=3)
    for c, k in enumerate(SENSE):
        for cells in groups[k].values():
            b.stimulate(cells, 0.0, fly=c, key=c)
    b.start([1], np.concatenate([c * b.n + np.r_[f['mnL'], f['mnR']] for c in range(3)]).astype(np.int32))
    pos = {e: i for i, e in enumerate(b.stim)}
    slot = {(c, g): np.array([pos[c * b.n + x] for x in groups[k][g]], np.int64) for c, k in enumerate(SENSE) for g in groups[k]}
    p0 = np.fromiter(b.stim.values(), np.float32, len(b.stim))
    sub = annotation_table().subclass.fillna('').to_numpy()
    mn = np.r_[f['mnL'], f['mnR']]
    legmn = {k: np.flatnonzero((sub[mn] == {'f': 'fl', 'm': 'ml', 'h': 'hl'}[k[1]]) & (np.arange(nc) < nl if k[0] == 'l' else np.arange(nc) >= nl)) for k in ls.LEGS}
    cum, tr, touch, qs = [np.zeros(3 * nc)], {k: [q0] for k in SENSE}, {k: [] for k in SENSE}, {k: q0 for k in SENSE}
    for i in range(STEPS):
        p = p0.copy()
        for c, k in enumerate(SENSE):
            r = feel[k](qs[k])
            touch[k].append([r[g, 'touch'] for g in ls.LEGS])
            if k != 'nosense':
                for g, hz in r.items():
                    p[slot[c, g]] = hz * b.dt
        p.tofile(b.work / 'stim_p.bin')
        cum.append(b.advance(K).astype(float))
        x = (cum[-1] - cum[-1 - min(i + 1, WINDOW)]).reshape(3, nc) * (WINDOW / min(i + 1, WINDOW))
        for c, k in enumerate(SENSE):
            eng[k].send(gait[k].ctrl(drive(f, x[c], nl) if i + 1 >= WINDOW else np.zeros(2)))
        for k in SENSE:
            qs[k] = eng[k].recv()
            tr[k].append(qs[k])
    b.close()
    for k in SENSE:
        eng[k].close()
    spk = np.diff(np.array(cum), axis=0).reshape(STEPS, 3, nc)[WINDOW:]
    bins = lambda a: a[:len(a) // 10 * 10].reshape(-1, 10, *a.shape[1:]).sum(1)
    spec = {}
    for c, k in enumerate(SENSE):
        t, s_ = bins(np.array(touch[k])[WINDOW:]), bins(spk[:, c])
        mnl = np.stack([s_[:, legmn[g]].sum(1) for g in ls.LEGS], 1)
        cc = np.nan_to_num(np.corrcoef(t.T, mnl.T)[:6, 6:])
        spec[k] = float(np.mean([cc[j, j] - np.delete(cc[j], j).mean() for j in range(6)]))
    r = {k: metrics(m, None, np.array(tr[k])) for k in SENSE}
    fw = {k: float(r[k]['forward'][-1]) for k in SENSE}
    print(json.dumps({'forward_mm': fw, 'tilt_sense': r['sense']['tilt_max'], 'specificity': spec, 'mn_spikes': {k: float(spk[:, c].sum()) for c, k in enumerate(SENSE)}}))
    ok = [(f"sense walks {fw['sense']:.2f} mm > {BODY_LENGTH}, tilt {r['sense']['tilt_max']:.1f} < 20", fw['sense'] > BODY_LENGTH and r['sense']['tilt_max'] < 20),
          (f"leg specificity sense {spec['sense']:+.3f} - nosense {spec['nosense']:+.3f} >= 0.05", spec['sense'] - spec['nosense'] >= 0.05),
          (f"mutant shuffled legs: sense {spec['sense']:+.3f} - shuffled {spec['shuffled']:+.3f} >= 0.05", spec['sense'] - spec['shuffled'] >= 0.05)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    (OUT / 'mcns-sense.json').write_text(json.dumps({'forward_mm': fw, 'specificity': spec}))
    if not all(v for _, v in ok):
        sys.exit('FAIL leg sense in the closed loop')
    print('PASS leg sense in the closed loop: each leg\'s touch reaches its own motor neurons')


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
    {'vision': vision, 'sense': sense}.get(sys.argv[1] if sys.argv[1:] else '', main)()
