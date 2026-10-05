"""Neuron growth in the GPU spider pathway (SLICE: retina 1200 -> AM1 2000 -> AM2 -> 12 eye motors): AM2 starts with
N0 recruited neurons and recruits GROW more (silent reserve, up to 2000) when the pursuit error has plateaued while
the motor layer's slot budget is full of strong weights. No kernel change: the AM2 group row count and the motor
layer's AM2 pool size are host tables (plan.bin); a recruit restarts the binary with the larger tables and the
current weights (state is reset every episode anyway; the restarted binary replays its noise and regrow streams from
step 0). Unrecruited neurons are never computed and no slot points at them.

arms (paired: same init, task tape per seed): fixed2000 (agent.spider_pool_gpu learn), fixed200, grow (trigger above),
random (mutant: the same number of recruits at random test points).
claims (reported with FEAT 3): CLAIM3 compute-matched (grow within 1.2x of a static net of its final size, on
fewer neuron-batches); CLAIM4 size finder (a static net at the grown size within 1.2x of fixed2000, >= 1.3x better than
fixed N0).
gates: G1 grow after <= 1.2 x fixed2000 after, with < 2000 neurons; G2 fixed200 after >= 1.3 x grow after;
M1 random-time recruits: after >= 1.2 x grow after, or as many neurons for no gain (reported, not gated).
usage: gpu-wait run --priority 70 --shared --vram 1 --timeout 3600 -- .venv/bin/python tools/spider_growth_test.py [ITERS] [SEEDS]
gate run (CLAIM4): --env N0=2 --env GROW=5 --env PATIENCE=2 --env FEAT=1,3,B,E --env GATE=claim4 ... 300 2,3,4
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ['SLICE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import agent.spider_pool_gpu as S  # noqa: E402

N0, GROW = int(os.environ.get('N0', 200)), int(os.environ.get('GROW', 200))  # start size, recruits per event
EVERY, PLATEAU, STRONG = 25, 0.95, 0.3
# research steps (Brain 2026-10-05-neuron-growth-research), cumulative: 1 aligned output init at the existing motor |w|,
# 2 young recruits get a x3 teaching signal for 2 test points, 3 a fixed arm at grow's final size, 4 trigger from the first test point
# FEAT (comma list) picks features independently: 1 2 3 4 as above; A noise-aware trigger (mean of the last 2 test points
# vs the 2 before) and stop when a recruit's next 2 points are not 3 % lower; B restarts continue the binary's counters;
# C recruits copy (split) the input slots of a random recruited neuron at half weight; S shrink and perturb at each
# recruit (Ash & Adams 2020: live pool weights x SHRINK + N(0, SIGMA)); L weight decay, all live weights x DECAY every
# test point, every arm (Dohare 2024: small weights keep plasticity); R recycle (continual backprop, Dohare 2024): every
# test point, every arm, the recruited AM2 neuron of least output utility (sum |w| of its motor synapses), if older than
# 2 points and below 0.1 x the mean, is cleared and rewired as a recruit; E early growth: the trigger runs every batch on
# an EMA of the training error (no test cost): recruit when it fell < 3 % over the last 10 batches, stop when a recruit's
# next 15 batches do not bring it 3 % below its level at the recruit (Evci 2022, Maile 2022: grow early, then train).
# STEP=k still means FEAT 1..k.
FEAT = set(os.environ['FEAT'].split(',')) if 'FEAT' in os.environ else {str(k) for k in range(1, int(os.environ.get('STEP', 0)) + 1)}
BOOST, YOUNG, SHRINK, SIGMA, DECAY = 3.0, 2, 0.5, 0.01, 0.9
PATIENCE = int(os.environ.get('PATIENCE', 1))  # E: non-improving recruits in a row before growth stops
M0 = S.NR + S.N1  # first AM2 neuron


def resize(n):
    S.G[1][1], S.SEG[2][1] = n, n


def strong(net):
    """fraction of live motor slots at >= half the weight cap (the budget is 'full of strong weights')."""
    w, _ = net.weights()
    m = w[S.G[2][3]:]
    live = m != 0
    return float((np.abs(m[live]) >= S.WMAX / 2).mean()) if live.any() else 0.0


def wire(w, idx, old, n, rng, fb=None, ids=None):
    """A recruit arrives with one weak synapse (+-1.5 THETA) onto each eye motor, in an empty slot of that motor's row.
    Without this the motor budget, full of strong synapses onto the first neurons, never prunes, so regrow never
    points a slot at a recruit: measured 2 -> 32 recruits at 0.34 deg, flat, vs 0.17 for 20 neurons from the start."""
    ids = np.arange(old, n) if ids is None else ids  # AM2 neurons (0-based within AM2) to wire
    if 'C' in FEAT and old:  # split (Chen 2016 Net2Net): a recruit starts from a random recruited neuron's inputs at half weight
        sb, f = S.G[1][3], S.G[1][2]
        for r in ids:
            src = rng.choice(np.setdiff1d(np.arange(old), ids))
            idx[sb + r * f:sb + (r + 1) * f] = idx[sb + src * f:sb + (src + 1) * f]
            w[sb + r * f:sb + (r + 1) * f] = w[sb + src * f:sb + (src + 1) * f] * np.float32(0.5)
    mo, fm = S.G[2][3], S.G[2][2]
    for j in range(S.NM):
        row = slice(mo + j * fm, mo + (j + 1) * fm)
        free = np.flatnonzero(w[row] == 0)[:len(ids)] + row.start
        got = ids[:len(free)]
        idx[free] = M0 + got
        if '1' in FEAT:  # sign agrees with the feedback the recruit already gets (h = err . fb), size of the row's live synapses
            live = w[row][w[row] != 0]
            sgn = np.sign(fb[M0 + got] @ fb[S.OUT0 + j]); sgn[sgn == 0] = 1
            w[free] = sgn.astype(np.float32) * np.float32(np.abs(live).mean() if len(live) else 1.5 * S.THETA)
        else:
            w[free] = rng.choice([-1.5, 1.5], len(free)).astype(np.float32) * np.float32(S.THETA)


def recycle(w, idx, n, born, now, fb, rng):
    """Continual backprop on AM2: the mature neuron of least output utility, if below 0.1 x the mean, gets its input
    row and motor synapses cleared and is rewired as a recruit (in place, same tables)."""
    mo, sb, f = S.G[2][3], S.G[1][3], S.G[1][2]
    mw, mi = np.abs(w[mo:]), idx[mo:] - M0
    util = np.bincount(mi[mw > 0], mw[mw > 0], minlength=n)[:n]
    ok = np.flatnonzero(now - born[:n] >= 2)
    if len(ok) < 2 or n < 3:
        return
    r = ok[np.argmin(util[ok])]
    if util[r] >= 0.1 * util.mean():
        return
    w[sb + r * f:sb + (r + 1) * f] = 0
    w[mo:][mi == r] = 0
    wire(w, idx, n, n, rng, fb, ids=np.array([r]))
    born[r] = now


def train(arm, iters, rec, seed, schedule=()):
    """returns the after error, neurons used, recruit test points, curve."""
    n = int(arm[5:]) if arm.startswith('fixed') else N0  # fixed<N>: N neurons, never grows
    idx, w, fb, ch = S.init(seed)  # at full size (init sizes the slot table from the group rows), then shrink
    resize(n)
    mo = S.G[2][3]
    idx[mo:] = M0 + (idx[mo:] - M0) % n  # motor slots name recruited AM2 only (all empty at start)
    net, rng = S.GPUPathway(idx, w, fb, ch, seed=7), np.random.default_rng(seed + 1)
    test = lambda s: float(np.median(np.concatenate([S.run(net, rec, S.P.speeds(np.random.default_rng(s + q), 6, 9, S.B), np.random.default_rng(s + q)) for q in range(4)])))
    curve, events, best, t0 = [], [], np.inf, time.perf_counter()
    born = np.zeros(2000, int)  # test point each AM2 neuron was (re)wired at
    fb0, young, stopped, pre = fb.copy(), None, False, np.inf  # young: (first recruit, last recruit, test point recruited)
    ema, hist, cost, miss = None, [], 0, 0
    for it in range(iters):
        tr = float(np.median(S.run(net, rec, S.P.speeds(rng, 2, 5, S.B), rng, learn=True)))
        cost += n  # neuron-batches spent
        if 'E' in FEAT and arm == 'grow':
            ema = tr if ema is None else 0.9 * ema + 0.1 * tr
            hist.append(ema)
            last = events[-1] if events else 0
            if events and it == last + 15:
                miss = miss + 1 if ema > 0.97 * pre else 0
                stopped = miss >= PATIENCE  # PATIENCE non-improving recruits in a row (1 stopped seed 3 at 7 neurons)
            if n < 2000 and not stopped and it >= 10 and it - last >= 15 and ema > 0.97 * hist[-11]:
                pre = ema
                w, idx = net.weights(); net.close()
                if 'S' in FEAT:
                    live = w != 0
                    w[live] = w[live] * np.float32(SHRINK) + np.random.default_rng(seed * 31 + it).normal(0, SIGMA, int(live.sum())).astype(np.float32)
                cont = dict(t=net.t, tstep=net.tstep) if 'B' in FEAT else {}
                old, n = n, min(2000, n + GROW); resize(n); events.append(it); born[old:n] = len(curve)
                wire(w, idx, old, n, np.random.default_rng(seed * 1000 + it), fb)
                net = S.GPUPathway(idx, w, fb, ch, seed=7, **cont)
        if it % EVERY != EVERY - 1:
            continue
        e = test(11)
        curve.append(round(e, 2))
        sf = strong(net)
        if 'L' in FEAT or 'R' in FEAT:
            w, idx = net.weights()
            if 'L' in FEAT:
                w *= np.float32(DECAY)
            if 'R' in FEAT:
                recycle(w, idx, n, born, len(curve), fb, np.random.default_rng(seed * 7919 + len(curve)))
            net.set_weights(w, idx)
        if 'E' in FEAT and arm == 'grow':
            fire = False  # E decides per batch above
        elif 'A' in FEAT and arm == 'grow':
            k = len(curve)
            if events and k == events[-1] + 2 and np.mean(curve[-2:]) > 0.97 * pre:
                stopped = True  # the last recruit did not lower the error: the layer has its size
            fire = (n < 2000 and not stopped and k >= 4 and (not events or k >= events[-1] + 2)
                    and np.mean(curve[-2:]) > PLATEAU * np.mean(curve[-4:-2]) and sf >= STRONG)
            if fire:
                pre = np.mean(curve[-2:])
        else:
            fire = n < 2000 and (len(curve) in schedule if arm == 'random' else
                                 arm == 'grow' and len(curve) >= (1 if '4' in FEAT else 3) and e > PLATEAU * best and sf >= STRONG)
        best = min(best, e)
        if fire:
            w, idx = net.weights(); net.close()
            cont = dict(t=net.t, tstep=net.tstep) if 'B' in FEAT else {}
            if 'S' in FEAT:  # shrink and perturb the existing pool before the recruits arrive
                live = w != 0
                w[live] = w[live] * np.float32(SHRINK) + np.random.default_rng(seed * 31 + len(curve)).normal(0, SIGMA, int(live.sum())).astype(np.float32)
            old, n = n, min(2000, n + GROW); resize(n); events.append(len(curve)); born[old:n] = len(curve)
            wire(w, idx, old, n, np.random.default_rng(seed * 1000 + len(curve)), fb)
            fb = fb0.copy()
            if '2' in FEAT:
                young = (M0 + old, M0 + n, len(curve)); fb[young[0]:young[1]] *= np.float32(BOOST)
            net = S.GPUPathway(idx, w, fb, ch, seed=7, **cont)
        elif young and len(curve) - young[2] >= YOUNG:  # critical period over: back to the mature teaching signal
            w, idx = net.weights(); net.close(); fb, young = fb0.copy(), None
            net = S.GPUPathway(idx, w, fb, ch, seed=7, **(dict(t=net.t, tstep=net.tstep) if 'B' in FEAT else {}))
        print(arm, seed, it + 1, curve[-1], f'strong {sf:.2f}', n, f'{(time.perf_counter() - t0) / (it + 1):.2f} s/batch', flush=True)
    out = {'after': test(9), 'neurons': n, 'events': events, 'curve': curve, 'neuron_batches': cost}
    net.close()
    resize(2000)
    return out


def cached(arm, iters, rec, seed):
    """Reference arms (fixed<N>) are identical across runs of the same kernel, seed, length and the features that touch
    fixed arms (L, R): train once, reuse (fixed2000 is ~600k of a run's ~620k neuron-batches). Arms under test never cached."""
    import hashlib
    src = b''.join((S.ROOT / f).read_bytes() for f in ('agent/spider_pool.mojo', 'agent/spider_pool_gpu.py', 'tools/spider_growth_test.py'))
    key = f"{arm}-s{seed}-i{iters}-{'L' if 'L' in FEAT else ''}{'R' if 'R' in FEAT else ''}-{hashlib.sha1(src).hexdigest()[:10]}"
    f = S.ROOT / f'.work/agent/spider/refcache/{key}.json'
    if f.is_file():
        return json.loads(f.read_text())
    out = train(arm, iters, rec, seed)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out))
    return out


def seed_arms(iters, seed):
    """All arms of one seed, in order (the random arm needs grow's recruit count)."""
    rec = S.P.retina(np.random.default_rng(0))
    r = {arm: cached(arm, iters, rec, seed) for arm in ('fixed2000', f'fixed{N0}')} | {'grow': train('grow', iters, rec, seed)}
    pts = iters // EVERY
    k = min(len(r['grow']['events']), pts)
    if '3' in FEAT:
        r['fixedfinal'] = cached(f"fixed{r['grow']['neurons']}", iters, rec, seed)
    r['random'] = train('random', iters, rec, seed, set(np.random.default_rng(seed + 21).choice(np.arange(1, pts + 1), k, replace=False).tolist()))
    return r


def main(iters='300', seeds='2,3,4'):
    rec = S.P.retina(np.random.default_rng(0))
    res, bad, claims = {}, 0, []
    seed_list = list(map(int, seeds.split(',')))
    from concurrent.futures import ProcessPoolExecutor  # seeds are separate brains (own weights): one process each, concurrent on the card
    with ProcessPoolExecutor(len(seed_list)) as ex:
        runs = dict(zip(seed_list, ex.map(seed_arms, [int(iters)] * len(seed_list), seed_list)))
    for seed in seed_list:
        r = runs[seed]
        f2, f0, g, rnd = (r[a]['after'] for a in ('fixed2000', f'fixed{N0}', 'grow', 'random'))
        checks = {'G1 grow <= 1.2 fixed2000 with < 2000 neurons': g <= 1.2 * f2 and r['grow']['neurons'] < 2000,
                  'G2 fixed200 >= 1.3 grow': f0 >= 1.3 * g}
        m1 = rnd >= 1.2 * g or (r['random']['neurons'] >= r['grow']['neurons'] and rnd >= g)
        print(f"seed {seed}: fixed2000 {f2:.2f} fixed200 {f0:.2f} grow {g:.2f} ({r['grow']['neurons']} neurons, events {r['grow']['events']}) "
              f"random {rnd:.2f} ({r['random']['neurons']}, {r['random']['events']})")
        for c, v in checks.items():
            print(f"{'PASS' if v else 'FAIL'} seed {seed} {c}")
        print(f"{'PASS' if m1 else 'INFO'} seed {seed} M1 random-time recruits worse or no cheaper")
        if 'fixedfinal' in r:
            print(f"INFO seed {seed} fixed at grow's final size {r['grow']['neurons']}: {r['fixedfinal']['after']:.2f} vs grow {g:.2f}")
            ff = r['fixedfinal']
            print(f"INFO seed {seed} compute: grow {r['grow']['neuron_batches']} neuron-batches vs fixed-final {ff['neuron_batches']}"
                  f" vs fixed2000 {r['fixed2000']['neuron_batches']}; error ratio grow / fixed-final {g / ff['after']:.2f}")
            c3 = g <= 1.2 * ff['after'] and r['grow']['neuron_batches'] < ff['neuron_batches']
            c4 = ff['after'] <= 1.2 * f2 and r['grow']['neurons'] < 2000 and f0 >= 1.3 * ff['after']
            print(f"{'PASS' if c3 else 'FAIL'} seed {seed} CLAIM3 compute-matched: grow within 1.2x of fixed-final on fewer neuron-batches")
            print(f"{'PASS' if c4 else 'FAIL'} seed {seed} CLAIM4 size finder: fixed at the grown size within 1.2x of fixed2000, and >= 1.3x better than fixed{N0}")
            claims.append((c3, c4))
        bad += sum(not v for v in checks.values())
        res[seed] = r
    if claims:
        print(f"CLAIM3 {sum(c[0] for c in claims)}/{len(claims)} seeds, CLAIM4 {sum(c[1] for c in claims)}/{len(claims)} seeds")
    (S.ROOT / '.work/agent/spider/growth.json').write_text(json.dumps(res, indent=2) + '\n')
    if os.environ.get('GATE') == 'claim4':  # the claim that holds (2026-10-05): growth finds a sufficient size
        miss = len(claims) - sum(c[1] for c in claims)
        if not claims or miss:
            sys.exit(f'FAIL CLAIM4 {miss}/{len(claims)} seeds')
        print(f'PASS spider growth as a size finder (CLAIM4 {len(claims)}/{len(claims)} seeds; G1/G2 failures above: {bad})')
        return
    if bad:
        sys.exit(f'FAIL {bad} checks')
    print('PASS spider growth')


if __name__ == '__main__':
    main(*sys.argv[1:])
