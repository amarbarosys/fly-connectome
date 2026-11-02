"""U6-bio: a walking rhythm from the MCNS nerve cord, by giving its interneurons a bursting mechanism (docs/plans/FULL_FLY.md U6).

Mechanism, no engine change: every vnc_intrinsic cell gets two self-edges on the slow modulation planes (ShiuLive.modulate,
as leg_mn.adapt_brain does): each spike adds W_SYN x SP to a fast depolarizing level (plane 0, tau TAU_P, b +1: a
spike-triggered plateau, the stand-in for a persistent inward current) and W_SYN x SA to a slow hyperpolarizing level
(plane 1, tau TAU_A, b -1: adaptation, the stand-in for a Ca-activated K / Na-pump current). The plateau makes a driven
cell fire a burst; the adaptation ends it and holds the cell silent until it decays. Alternation between antagonists must
come from the connectome (13A / 13B inhibitory premotor neurons, Syed et al. 2026). MNs, sensory and brain cells unchanged.
  python -m agent.body_engine.cpg_bio cpu                 CPU: one cell bursts, a half-center alternates, its no-inhibition mutant does not
  gpu-wait ... -- .venv/bin/python -m agent.body_engine.cpg_bio fit SEED SPEC TP,TA [TP,TA ...]   FIT sweep (SP x SA per column), SPEC cells:mode:het
  gpu-wait ... -- .venv/bin/python -m agent.body_engine.cpg_bio gate SEED                    G1 real VNC, G2 shuffled VNC
  gpu-wait ... -- .venv/bin/python -m agent.body_engine.cpg_bio g3                           G3: mechanism off == main, 200 steps
"""
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent import mcns, seeds

TAU_P, TAU_A, SP, SA = 0.01, 0.03, 160.0, 40.0  # s, s, plateau and adaptation scale per spike (x W_SYN mV); frozen on FIT 605 (FULL_FLY.md U6)
SPEC = 'exc:add:0'  # bio_brain: which cells (excitatory vnc_intrinsic), plateau mode, heterogeneity; frozen on FIT 605
VNC = ('vnc_intrinsic', 'vnc_motor', 'vnc_efferent', 'ascending_neuron')  # G2: edges with both ends in these are shuffled
SHUFFLE_SEED = 0
STEPS, BIN, REPEATS = 1000, 10, 2  # 1 ms control steps (K = 10 ticks each); 10 ms bins; FIT repeats per setting
OUT = Path('.work/agent/u6')
ROOT_MAIN = Path(subprocess.run(['git', 'worktree', 'list', '--porcelain'], capture_output=True, text=True).stdout.split()[1])  # the main checkout


def table():
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                        columns=['status', 'superclass', 'subclass', 'somaSide', 'type', 'trumanHl'])
    return a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)


def tibia(a, leg='lf'):
    """(flexor, extensor) MN ids of one leg: Ti flexor + Acc. ti flexor vs Ti extensor (as auto_fly)."""
    sc, sub, ty = (a[c].fillna('').to_numpy() for c in ('superclass', 'subclass', 'type'))
    m = (sc == 'vnc_motor') & (sub == f'{leg[1]}l') & (a.somaSide.fillna('').str[:1].to_numpy() == leg[0].upper())
    return np.flatnonzero(m & np.isin(ty, ['Ti flexor MN', 'Acc. ti flexor MN'])), np.flatnonzero(m & (ty == 'Ti extensor MN'))


def graph(shuffle=False):
    """(pre, post, w) of the MCNS Shiu graph; shuffle: VNC-internal edges' posts permuted (degree-preserving, SHUFFLE_SEED)."""
    from agent import shiu_g1
    from agent.shiu_shuffle import shuffle_edges
    pre, post, w = next(shiu_g1.arms(mcns.graph(), 'mcns'))[1]
    if shuffle:
        vnc = np.isin(table().superclass.fillna('').to_numpy(), VNC)
        e = vnc[pre] & vnc[post]
        post = post.copy()
        post[e] = shuffle_edges(pre[e], post[e], SHUFFLE_SEED)[0]
    return pre, post, w


def with_bursts(pre, post, w, cells, het=0.0):
    """Append two self-edges per burst cell (plateau, then adaptation); MCNS has no autapses. het: the plateau self-edge
    weight is exp(het x N(0, 1)) per cell (fixed draw, seed 0), so cells burst at different rates."""
    assert not (pre == post).any()
    wp = np.exp(het * np.random.default_rng(0).standard_normal(len(cells))).astype(w.dtype)
    return np.r_[pre, cells, cells], np.r_[post, cells, cells], np.r_[w, wp, np.ones(len(cells), w.dtype)]


def burst_cells(which, pre=None, w=None):
    """vnc_intrinsic cells given the mechanism: all, inh (net outgoing synapse count < 0) or exc (> 0); a '+mn' suffix
    adds the leg MNs (measured only: U6 registers interneurons)."""
    a = table()
    cells = np.flatnonzero(a.superclass.to_numpy() == 'vnc_intrinsic')
    base, mn = which.split('+')[0], which.endswith('+mn')
    if base != 'all':
        net = np.bincount(pre, w.astype(np.float64), minlength=len(a))[cells]
        cells = cells[net < 0] if base == 'inh' else cells[net > 0]
    if mn:
        cells = np.sort(np.r_[cells, np.flatnonzero((a.superclass == 'vnc_motor').to_numpy() & a.subclass.isin(['fl', 'ml', 'hl']).to_numpy())])
    return cells


def groups(b, cells):
    """edge group per CSR edge: 0 synapse, 1 plateau self-edge, 2 adaptation self-edge (stable sort keeps appended order)."""
    epre = np.repeat(np.arange(b.n), np.diff(b.row))
    s = np.flatnonzero(b.col == epre)
    assert len(s) == 2 * len(cells)
    g = np.zeros(len(b.col), np.uint8)
    g[s[0::2]], g[s[1::2]] = 1, 2
    return g


def bio_brain(seed, scales, tau=(TAU_P, TAU_A), shuffle=False, spec=None):
    """ShiuLive with the burst mechanism on vnc_intrinsic cells, forward DNs 80 Hz in every column.
    scales: per column (plateau scale, adaptation scale); (0, 0) is the mechanism present but silent.
    spec 'cells:mode:het' (default SPEC): cells all / inh / exc; mode add (plateau is a depolarizing current) or gain
    (plateau multiplies the cell's synaptic conductance, excitatory and inhibitory alike: 1 + level); het as with_bursts."""
    from agent.body_engine.mcns_link import FWD_HZ
    from agent.mcns_mapping import descending_name_sets
    from agent.shiu import ShiuLive
    which, mode, het = (spec or SPEC).split(':')
    pre, post, w = graph(shuffle)
    cells = burst_cells(which, *graph(False)[::2])  # the subset from the real wiring, also under the shuffle
    b = ShiuLive(*with_bursts(pre, post, w, cells, float(het)), n=mcns.graph()[4][0], flies=len(scales), seed=seed)
    b.edge_groups(groups(b, cells), np.array([[1.0, sp, sa] for sp, sa in scales], np.float32))
    b.modulate([-1, 0, 1], tau, (0.0, 0.0) if mode == 'add' else (1.0, 0.0), (1.0, -1.0) if mode == 'add' else (0.0, -1.0))
    dn, _ = descending_name_sets()
    for c in range(len(scales)):
        b.stimulate(dn['forward'], FWD_HZ, fly=c, key=c)
    return b


def measure(F, E, counts, steps=STEPS):
    """counts (steps, cells) of the flexor then extensor MNs -> corr of 10 ms bin sums, flexor 3-20 Hz power fraction, rates."""
    k = steps // BIN * BIN
    fx = counts[:k, :len(F)].sum(1).reshape(-1, BIN).sum(1).astype(float)
    ex = counts[:k, len(F):len(F) + len(E)].sum(1).reshape(-1, BIN).sum(1).astype(float)
    f = np.fft.rfftfreq(len(fx), BIN * 1e-3)
    P = np.abs(np.fft.rfft(fx - fx.mean())) ** 2
    r = float(np.corrcoef(fx, ex)[0, 1]) if fx.std() and ex.std() else float('nan')
    return {'corr': round(r, 3), 'power_3_20': round(float(P[(f >= 3) & (f <= 20)].sum() / max(P[1:].sum(), 1e-12)), 3),
            'peak_hz': float(f[1:][P[1:].argmax()]) if P[1:].any() else 0.0,
            'flexor_hz': round(float(fx.sum() / len(F) / (k * 1e-3)), 1), 'extensor_hz': round(float(ex.sum() / len(E) / (k * 1e-3)), 1)}


def run(b, readout, steps=STEPS):
    """start, advance steps x 10 ticks, per-step counts (steps, columns, cells)."""
    from agent.body_engine.fly_link import K
    C = b.flies
    b.start([1], np.concatenate([c * b.n + readout for c in range(C)]).astype(np.int32))
    cum, out = np.zeros(C * len(readout)), []
    for _ in range(steps):
        x = b.advance(K).astype(np.int64)
        out.append((x - cum).reshape(C, len(readout)))
        cum = x
    b.close()
    return np.array(out, np.int32)


def readout_cells(a):
    F, E = tibia(a)
    sc, hl = a.superclass.fillna('').to_numpy(), a.trumanHl.fillna('').to_numpy()
    vi = np.flatnonzero(sc == 'vnc_intrinsic')
    sample = np.random.default_rng(0).choice(vi, 2000, replace=False)
    return F, E, sample, np.r_[F, E, sample]


def describe(counts, F, E, sample, steps=STEPS):
    """per column: the G1 measure plus the vnc_intrinsic sample's firing (fraction > 0.5 Hz, mean Hz) and burstiness
    (fraction of the sample's spikes falling in the busiest 25 % of each cell's 10 ms bins)."""
    out = []
    for c in range(counts.shape[1]):
        m = measure(F, E, counts[:, c], steps)
        s = counts[:steps // BIN * BIN, c, len(F) + len(E):].reshape(-1, BIN, len(sample)).sum(1)
        hz = s.sum(0) / (steps * 1e-3)
        act = s[:, hz > 5]
        top = np.sort(act, 0)[-max(1, len(act) // 4):].sum() / max(act.sum(), 1)
        m |= {'vnc_firing': round(float((hz > 0.5).mean()), 3), 'vnc_mean_hz': round(float(hz.mean()), 1), 'vnc_top25_share': round(float(top), 3)}
        out.append(m)
    return out


def fit(seed, spec, *taus):
    """FIT sweep: one run per TP,TA (s); columns = (0, 0) then SP x SA, each setting twice (independent Poisson draws:
    a column's own key). Writes per-setting measures and the flexor / extensor / sample counts (uint8) per run."""
    seed = int(seed)
    seeds.claim('u6_bio_fit' if spec == 'all:add:0' else f'u6_bio_fit_{spec}', [seed])
    a = table()
    F, E, sample, R = readout_cells(a)
    sps = (40.0, 80.0, 160.0, 240.0, 320.0) if ':add:' in spec else (0.5, 1.0, 2.0, 4.0, 8.0)
    sas = (2.5, 5.0, 10.0, 20.0, 40.0)
    grid = [(0.0, 0.0)] + list(itertools.product(sps, sas))
    cols = grid * REPEATS
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for tt in taus:
        tau = tuple(float(x) for x in tt.split(','))
        t0 = time.perf_counter()
        counts = run(bio_brain(seed, cols, tau, spec=spec), R)
        sec = time.perf_counter() - t0
        d = describe(counts, F, E, sample)
        for i, (sp, sa) in enumerate(grid):
            ms = d[i::len(grid)]
            m = {k: round(float(np.mean([x[k] for x in ms])), 3) for k in ms[0]} | {'corr_each': [x['corr'] for x in ms]}
            res[f'{spec} {tau[0]}/{tau[1]} {sp:g}/{sa:g}'] = m
            print(f'{spec} tau {tau[0]}/{tau[1]} plateau {sp:g} adapt {sa:g}:', json.dumps(m), flush=True)
        np.savez_compressed(OUT / f'fit-{seed}-{spec.replace(":", "_")}-{tau[0]}-{tau[1]}.npz', counts=np.minimum(counts, 255).astype(np.uint8), cols=np.array(cols), F=F, E=E, sample=sample)
        print(f'{sec:.0f} s for {STEPS} steps x {len(cols)} columns', flush=True)
    (OUT / f'fit-{seed}-{spec.replace(":", "_")}.json').write_text(json.dumps(res, indent=1))


def gate(seed):
    """G1 real VNC, G2 shuffled VNC; both with the frozen mechanism, forward DNs 80 Hz, 1 s, no MN stimulus."""
    seed = int(seed)
    seeds.claim('u6_bio_gate', [seed])
    a = table()
    F, E, sample, R = readout_cells(a)
    res = {}
    for name, sh in (('real', False), ('shuffled', True)):
        t0 = time.perf_counter()
        counts = run(bio_brain(seed, [(SP, SA)], shuffle=sh), R)
        res[name] = describe(counts, F, E, sample)[0] | {'seconds': round(time.perf_counter() - t0, 1)}
        print(name, json.dumps(res[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'gate-{seed}.json').write_text(json.dumps(res, indent=1))
    g1, g2 = res['real'], res['shuffled']
    ok = [(f"G1 real: flexor / extensor corr {g1['corr']:+.3f} <= -0.5 and flexor 3-20 Hz power {g1['power_3_20']:.3f} >= 0.5",
           g1['corr'] <= -0.5 and g1['power_3_20'] >= 0.5),
          (f"G2 shuffled VNC: corr {g2['corr']:+.3f} > -0.2", not g2['corr'] <= -0.2)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL U6-bio gate')
    print('PASS U6-bio G1/G2: the nerve cord alternates the left front tibia flexor and extensor')


def g3():
    """G3: with the mechanism off (cpg_bio not used: mcns_link.brain), a fixed-seed 200-step run's MN counts are
    byte-equal to the same run from main's agent/ (git archive into .work/u6/main, .work and data/ reached through symlinks)."""
    main_dir = Path('.work/u6/main').resolve()
    if main_dir.exists():
        subprocess.run(['rm', '-rf', str(main_dir)], check=True)
    main_dir.mkdir(parents=True)
    subprocess.run(f'git archive main agent | tar -x -C {main_dir}', shell=True, check=True)
    for d in ('.work', 'data'):
        (main_dir / d).symlink_to(Path(d).resolve() if d == '.work' else ROOT_MAIN / d)  # data/: the main checkout's (the worktree lacks untracked annotation files)
    code = ("import sys, numpy as np\n"
            "from agent.body_engine.mcns_link import brain\n"
            "b, f, nl, nc = brain(seed=11)\n"
            "x = np.array([b.advance(10) for _ in range(200)], np.int32)\n"
            "b.close()\n"
            "x.tofile(sys.argv[1])\n")
    outs = []
    for name, root in (('branch', Path('.').resolve()), ('main', main_dir)):
        p = OUT.resolve() / f'g3-{name}.bin'
        OUT.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-c', code, str(p)], cwd=root, check=True, env={**__import__('os').environ, 'PYTHONPATH': str(root)})
        outs.append(p.read_bytes())
    same = outs[0] == outs[1]
    print(f'{"PASS" if same else "FAIL"} G3: mechanism off, 200 steps x 3 columns, {len(outs[0]) // 4} cumulative MN counts, '
          f'branch {"==" if same else "!="} main, {int(np.frombuffer(outs[0], np.int32)[-len(outs[0]) // 800:].sum())} spikes at step 200')
    if not same:
        sys.exit('FAIL G3')


def cpu():
    """CPU (ShiuBrain toy, frozen constants): one driven cell bursts with the mechanism and not without; two pools of 10
    burst cells with cross inhibition alternate (corr <= -0.4) and the no-inhibition mutant does not (corr > 0)."""
    from agent.shiu import ShiuBrain
    M, NIN, HZ, KIN = 10, 40, 80.0, 6.0
    rng = np.random.default_rng(0)
    A, B = NIN + np.arange(M), NIN + M + np.arange(M)
    pre, post, w, g = [], [], [], []
    for c in np.r_[A, B]:
        s = rng.choice(NIN, 20, replace=False)
        pre += [*s, c, c]; post += [c] * 22; w += [KIN] * 20 + [1.0, 1.0]; g += [0] * 20 + [1, 2]
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for q in Q:
                pre.append(p); post.append(q); w.append(-1.0); g.append(3)
    pre, post, w, g = map(np.array, (pre, post, w, g))
    cols = [(1, 5.0), (1, 0.0), (0, 0.0), (0, 5.0)]  # (mechanism, cross inhibition count)
    b = ShiuBrain(pre, post, w, n=NIN + 2 * M, flies=len(cols), seed=2)
    b.edge_groups(g[np.argsort(pre, kind='stable')].astype(np.uint8), np.array([[1.0, SP * m, SA * m, i] for m, i in cols], np.float32))
    b.modulate([-1, 0, 1, -1], (TAU_P, TAU_A), (0.0, 0.0), (1.0, -1.0))
    b.stimulate(np.arange(NIN), HZ)
    cnt = np.diff(np.array(b.run(list(range(100, 10001, 100)))), axis=0, prepend=0)  # (10 ms bins, cells, columns)
    f = np.fft.rfftfreq(cnt.shape[0], 0.01)
    pw = lambda x: float((np.abs(np.fft.rfft(x - x.mean())) ** 2)[(f >= 3) & (f <= 20)].sum() / max((np.abs(np.fft.rfft(x - x.mean())) ** 2)[1:].sum(), 1e-12))
    one = {k: pw(cnt[:, A[0], j].astype(float)) for j, k in ((1, 'on'), (2, 'off'))}
    r = {k: float(np.corrcoef(cnt[:, A, j].sum(1), cnt[:, B, j].sum(1))[0, 1]) for j, k in ((0, 'halfcenter'), (1, 'no_inhibition'))}
    print(json.dumps({'single_cell_3_20Hz_power': one, 'pool_corr': r, 'rates_hz': cnt[:, A].sum((0, 1)).tolist()}))
    ok = [(f"single cell bursts: 3-20 Hz power on {one['on']:.2f} >= 0.5, off {one['off']:.2f} < 0.5 (G1's bar)", one['on'] >= 0.5 > one['off']),
          (f"half-center alternates: corr {r['halfcenter']:+.2f} <= -0.4", r['halfcenter'] <= -0.4),
          (f"mutant without cross inhibition: corr {r['no_inhibition']:+.2f} > 0", r['no_inhibition'] > 0)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit('FAIL cpg_bio cpu')
    print('PASS cpg_bio cpu: the mechanism bursts and a half-center alternates')


if __name__ == '__main__':
    {'cpu': cpu, 'fit': fit, 'gate': gate, 'g3': g3}[sys.argv[1]](*sys.argv[2:])
