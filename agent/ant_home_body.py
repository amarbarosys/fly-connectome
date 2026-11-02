"""ANT A8: the 89/128 CX in the ant body (docs/plans/ANT.md). CPU MuJoCo body, one homing decision per run.
Outbound: compass P-steering along paths7 legs; each leg's measured chord (heading, length) is the brain's input.
Brain: m_trials with A7d gains decodes one home direction. Return: compass P-steering to that direction, T = outbound
path time. Not continuous CX control: the CX decides once, a scripted compass controller holds the course.
  python -m agent.ant_home_body smoke          body only, 8 ants
  gpu-wait run ... python -m agent.ant_home_body gate SEED
"""
import json
import multiprocessing as mp
import sys

import mujoco
import numpy as np

from agent import ant_cx as cx, seeds
from agent.ant_body import build
from agent.body_engine.ant_link import gait
from agent.body_engine.spider_link import yaw_of

LEG_MM, STEP = 40.0, 10  # mm at speed 1 per leg; control every STEP physics steps
MODEL = None


def model():
    global MODEL
    if MODEL is None: MODEL = mujoco.MjModel.from_xml_path(str(build()))
    return MODEL


def steer(m, d, target, t, secs=None, mm=None, nest=None):
    """Walk on compass target (deg) until secs or mm of path. Returns (t, closest distance to nest)."""
    p0, path, best, end = d.qpos[:2].copy(), 0.0, np.inf, t + (secs or 30.0)
    while t < end and (mm is None or path < mm):
        last = d.qpos[:2].copy()
        e = (target - yaw_of(d.qpos) + 180) % 360 - 180
        d.ctrl[:] = gait(np.array(t), np.array([1.0 if abs(e) < 15 else 0.6]), np.array([np.clip(e / 15.0, -1, 1)]))[0]
        for _ in range(STEP): mujoco.mj_step(m, d)
        t += STEP * m.opt.timestep
        if abs(e) < 15: path += float(np.linalg.norm(d.qpos[:2] - last))
        if nest is not None: best = min(best, float(np.linalg.norm(d.qpos[:2] - nest)))
    return t, best


def outbound(job):
    h, u = job
    m = model(); d = mujoco.MjData(m)
    for _ in range(2000): mujoco.mj_step(m, d)
    nest, t, pts = d.qpos[:2].copy(), 0.0, []
    for hh, uu in zip(h, u):
        s = d.qpos[:2].copy()
        t, _ = steer(m, d, hh, t, mm=uu * LEG_MM)
        pts.append(d.qpos[:2] - s)
    pts = np.array(pts)
    return d.qpos.copy(), d.qvel.copy(), t, nest, np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360, np.linalg.norm(pts, axis=1) / LEG_MM


def homing(job):
    q, v, t, nest, target, length = job
    m = model(); d = mujoco.MjData(m); d.qpos[:], d.qvel[:] = q, v
    start = float(np.linalg.norm(q[:2] - nest))
    _, best = steer(m, d, target, t, mm=2 * length, nest=nest)
    return best / max(start, 1e-9), start


def run_body(pool, h, u):
    r = pool.map(outbound, list(zip(h, u)))
    H, U = np.array([x[4] for x in r]), np.array([x[5] for x in r])
    return r, H, np.clip(U / 2.41, 0.0, 1.0)  # 2.41: smoke chord / scripted length, brain trained on u <= 1


def setup():
    cx.M_EXC, cx.SUB, cx.M_PRE = True, cx.cx_types(), cx.A7D['pre']


def score(pool, name, r, dec, home):
    bad = np.isnan(dec)
    rng = np.random.default_rng(0)
    target = np.where(bad, rng.uniform(0, 360, len(dec)), dec)  # no decision: a random direction
    res = pool.map(homing, [(x[0], x[1], x[2], x[3], t, 2 * LEG_MM * 3) for x, t in zip(r, target)])
    near = np.array([a for a, _ in res])
    e = cx.e6(dec, home)
    print(f'{name}: median heading err {np.median(e):.1f} deg, hits<30 {int((e < 30).sum())}/{len(e)}, median closest/start {np.median(near):.3f}, within 20% {int((near <= .2).sum())}', flush=True)
    return {'median_err': float(np.median(e)), 'near': float(np.median(near)), 'within20': int((near <= .2).sum())}


def smoke():
    rng = np.random.default_rng(1)
    h, u, d = cx.paths7(rng, 8)
    with mp.Pool(8) as p:
        r, H, U = run_body(p, h, u)
        home = np.array([np.degrees(np.arctan2(*(x[3] - x[0][:2])[::-1])) % 360 for x in r])
        print('chord heading err', np.round(np.median(cx.err(H, h)), 1), 'length ratio', np.round(np.median(U / u), 2), 'time', np.round(np.median([x[2] for x in r]), 1))
        score(p, 'oracle home', r, home, home)


def gate(seed, k=32):
    seed = int(seed)
    seeds.claim('ant_a8_gate', [seed])
    setup()
    z = np.load(cx.mcns.ROOT / '.work/agent/ant/a7d-gains.npz')
    gain, b = 2.0 ** z['real'], cx.A7D['b']
    rng = np.random.default_rng(seed)
    fit, test = cx.paths7(rng, k), cx.paths7(rng, k)
    with mp.Pool(14) as p:
        r, H, U = run_body(p, *test[:2])
        home = np.array([np.degrees(np.arctan2(*(x[3] - x[0][:2])[::-1])) % 360 for x in r])
        cols = [(a, c, None) for a, c in zip(*fit[:2])] + [(a, c, None) for a, c in zip(H, U)]
        cols += [(a, np.full(3, .6), k + i) for i, a in enumerate(H)]
        _, x, _, types, phase, fb = cx.m_trials(cols, None, seed, b, gain=gain)
        best = None
        for cand in cx.HOME_READ:
            for sg in (1, -1):
                raw = cx.dec6(x[:k], types, phase, fb, cand, sg)
                o = float(np.degrees(np.angle(np.nanmean(np.exp(1j * np.radians(raw - fit[2]))))))
                e = float(np.median(cx.e6((raw - o) % 360, fit[2])))
                if np.isfinite(o) and (best is None or e < best[0]): best = (e, cand, sg, o)
        e, cand, sg, o = best
        print(f'decoder {cand} s {sg:+d} off {o:.1f}, FIT median {e:.1f}', flush=True)
        dec = lambda xx: cx.dec6(xx, types, phase, fb, cand, sg, o)
        out = {'oracle': score(p, 'oracle', r, home, home), 'real': score(p, 'real', r, dec(x[k:2 * k]), home),
               'no odometer': score(p, 'no odometer', r, dec(x[2 * k:]), home)}
        xs = cx.m_trials([(a, c, None) for a, c in zip(H, U)], 5, seed, b, gain=gain)[1]
        out['shuffled'] = score(p, 'shuffled CX', r, dec(xs), home)
    ok = out['real']['median_err'] < 30 and out['real']['near'] <= .2 and all(out[a]['median_err'] > 60 for a in ('no odometer', 'shuffled'))
    print('PASS A8' if ok else 'FAIL A8')
    (cx.mcns.ROOT / f'.work/agent/ant/a8-{seed}.json').write_text(json.dumps(out))


if __name__ == '__main__':
    model()  # once, before the pool forks: build() rewrites the meshes
    {'smoke': smoke, 'gate': gate}[sys.argv[1]](*sys.argv[2:])
