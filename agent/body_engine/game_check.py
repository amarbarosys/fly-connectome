"""Game-state gates for the engine step (-DGAME=1, STAGE 7; private note GAME_LAYER): energy, health, stamina, dead per animal, per physics step.

  python -m agent.body_engine.game_check build [fly|spider]
  gpu-wait run --priority 60 --timeout 1800 -- .venv/bin/python -m agent.body_engine.game_check check [fly|spider]

Per body (fly: the standing fly, spider: models/spider.py), the engine runs live.mojo -DBRAIN=1 (host ctrl, K=1, no render) under one
perturbed ctrl stream and writes its game rows after every step (-DPROFILE=1 trace). MuJoCo steps the same stream; its per-step
work, sum |qfrc_actuator x qvel before the step| x dt, drives game_events.replay with the same rules (the oracle).
  (1) work: energy and stamina rows == the oracle's after every one of N steps (bar: 1e-3 of the drain, the float32 force x velocity floor)
  (2) death: basal drain so the animal dies at step 5: dead == 1 from the oracle's step on, energy frozen at the oracle's value, and
      qfrc_actuator and actuator_force exactly 0 from the step after (servos and adhesion off), nonzero while alive
check.sh game adds the mutants: the work term skipped, the dead gate removed.
"""
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from agent.body_engine import live, spectate_check as sc
from kinestride.game_events import Animal, Event, Rules, replay
from kinestride.models.spider import build as spider_build
from kinestride.render_check import MJCF

OUT = Path('.work/agent/body_engine/game')
N = 40
BODIES = {'fly': lambda: MJCF, 'spider': lambda: str(spider_build())}
CAM = ['0', '0', '10', '0', '0', '0', '45']


def prepare(name):
    d = OUT / name
    cam, m = live.prepare(d, mjcf=BODIES[name](), dist=15.0 if name == 'spider' else live.DIST)
    np.array([1, 2.0], np.float32).tofile(d / 'roots.bin')
    (d / 'mf.bin').replace(d / 'mf.base')  # the rules ride at the end of mf.bin (smooth.GAMEF); live.prepare re-exports every run, so the fresh export is the base
    return d, m


def ctrl_stream(m, d, steps=N):
    """Per step ctrl (steps, NU): the holding ctrl plus a sine on every joint servo (adhesion held), inside the ctrlrange."""
    u0 = np.fromfile(d / 'ctrl.bin', np.float32).astype(np.float64)
    joint = m.actuator_trntype != mujoco.mjtTrn.mjTRN_BODY
    u = u0 + joint * 0.15 * np.sin(0.4 * np.arange(steps)[:, None] + np.arange(len(u0)))
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    return np.where(m.actuator_ctrllimited, np.clip(u, lo, hi), u).astype(np.float32)


def set_rules(d, basal, work, regen=0.0):
    np.r_[np.fromfile(d / 'mf.base', np.float32), np.float32([basal, work, regen])].astype(np.float32).tofile(d / 'mf.bin')


def engine(d, u, frames=None):
    """N steps under ctrl stream u (K=1, BRAIN=1): rows (energy, health, stamina, dead, sum |qfrc_actuator|, sum |actuator_force|) per step.
    frames: a file path = also draw every step with the HUD (-DHUD=1, frames on fd 4), else no render."""
    b = sc.build(d, 'game', ['-DPROFILE=1', '-DGAME=1'] + (['-DHUD=1'] if frames else []), 1, brain=1)
    argv = [str(b), str(d) + '/', str(len(u)), *CAM, '1' if frames else '0', '0', str(OUT / 'roots.f32'), str(OUT / 'game.f32')]
    if frames:
        argv = ['bash', '-c', 'exec "$0" "$@" 4>"$FRAMES"', *argv]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env={**os.environ, 'FRAMES': str(frames)})
    for row in u:
        p.stdin.write(np.ascontiguousarray(row.reshape(-1, 1), np.float32).tobytes())
        p.stdin.flush()
    p.stdin.close()
    err = p.stderr.read().decode()
    if p.wait():
        raise SystemExit(f'FAIL engine exited {p.returncode}: {err[-300:]}')
    return np.fromfile(OUT / 'game.f32', np.float32).reshape(-1, 6)


def mujoco_work(m, d, u):
    """Per step sum |qfrc_actuator x qvel before the step| x dt: MuJoCo on the same start state and ctrl stream (engine's noslip 0)."""
    s = mujoco.MjData(m)
    s.qpos[:] = np.fromfile(d / 'qpos.bin', np.float32).astype(np.float64)
    s.qvel[:] = np.fromfile(d / 'qvel.bin', np.float32).astype(np.float64)
    work = []
    for row in u:
        s.ctrl[:] = row
        v0 = s.qvel.copy()
        mujoco.mj_step(m, s)
        work.append(float(np.abs(s.qfrc_actuator * v0).sum() * m.opt.timestep))
    return np.array(work)


def oracle(work, rules):
    """Rows (energy, health, stamina, dead) after every step of game_events.replay."""
    animals, out = [Animal()], []
    for w in work:
        animals = replay(animals, [Event(0, work=float(w))], {}, rules)
        out.append(animals[0].row())
    return np.array(out, np.float32)


def hud_check(name, d, u, state):
    """The HUD bars of the last frame show the engine's own rows: fill pixels == round(row x width), grey when dead (HUD pixels alpha 254)."""
    g = sc.hud_geometry()
    engine(d, u, OUT / 'hud.rgba')
    img = np.fromfile(OUT / 'hud.rgba', np.uint8).reshape(-1, sc.H, sc.W, 4)[-1]
    row = np.fromfile(OUT / 'game.f32', np.float32).reshape(-1, 6)[-1]
    ok = True
    for i in range(3):
        y = g['BAR_Y'] + g['BAR_PITCH'] * i + g['BAR_H'] // 2
        px = img[y, g['BAR_X']:g['BAR_X'] + g['BAR_W']]
        fill = ~(px[:, :3] == 0x28).all(1)
        want = int(round(float(row[i]) * g['BAR_W']))
        good = int(fill.sum()) == want and (px[:, 3] == 254).all() and bool(fill[:want].all()) and (bool((px[fill][:, :3] == 0x80).all()) if row[3] else True)
        ok &= bool(good)
        print(('PASS' if good else 'FAIL'), f'{name} HUD {state} bar {i}: {int(fill.sum())} px, engine row {row[i]:.4f} x {g["BAR_W"]} = {want}{", grey" if row[3] else ""}')
    return ok


def check(name):
    d, m = prepare(name)
    u = ctrl_stream(m, d)
    work = mujoco_work(m, d, u)
    print(f'{name}: MuJoCo work per step {work.mean():.3e} mean, {work.sum():.3e} total over {N} steps')
    rules = lambda basal, rate: Rules(basal_drain=basal, work_drain=rate, health_regen=0.0, rest_recovery=0.0, hazard_damage=0.0, resource_energy=0.0)
    # (1) work drain: the work term takes ~0.4 of the energy over N steps and the basal ~0.1, both exercised
    basal, rate = 0.1 / N, 0.4 / work.sum()
    set_rules(d, basal, rate)
    got = engine(d, u)[:, :4]
    want = oracle(work, rules(basal, rate))
    drain = 1 - want[-1, 0]
    err = float(np.abs(got[:, [0, 2]] - want[:, [0, 2]]).max() / drain)
    one = err <= 1e-3 and (got[:, 3] == 0).all() and hud_check(name, d, u, 'alive')
    print(('PASS' if one else 'FAIL'), f'{name} (1) work: energy/stamina vs game_events.replay on MuJoCo work, {N} steps: max err {err:.2e} of the drain {drain:.3f} (bar 1e-3); '
          f'engine energy {got[-1, 0]:.5f} stamina {got[-1, 2]:.5f}, oracle {want[-1, 0]:.5f} {want[-1, 2]:.5f}')
    # (2) death at step 5: servos and adhesion off from the next step
    set_rules(d, 0.2, rate)
    full = engine(d, u)
    got, act = full[:, :4], full[:, 4:]
    want = oracle(work, rules(0.2, rate))
    died = int(np.argmax(want[:, 3] > 0))
    flag = (got[:, 3] == want[:, 3]).all() and float(np.abs(got[died:, 0] - want[died:, 0]).max()) < 1e-6
    off = (act[died + 1:] == 0).all() and (act[:died].max(axis=1) > 0).all()
    two = bool(flag and off) and hud_check(name, d, u, 'dead')
    print(('PASS' if two else 'FAIL'), f'{name} (2) death at step {died}: dead flag == oracle on all {N} rows, energy frozen at {got[-1, 0]:.4f}; '
          f'sum |qfrc_actuator| alive {act[:died, 0].min():.3e}..{act[:died, 0].max():.3e}, after death {np.abs(act[died + 1:]).max():.3e} (must be 0)')
    return bool(one and two)


def main(cmd, *names):
    OUT.mkdir(parents=True, exist_ok=True)
    names = names or ('fly', 'spider')
    if cmd == 'build':
        for n in names:
            d, _ = prepare(n)
            set_rules(d, 0.0, 0.0)
            sc.build(d, 'game', ['-DPROFILE=1', '-DGAME=1'], 1, brain=1)
        return
    res = [check(n) for n in names]
    if not all(res):
        sys.exit('FAIL game: ' + ' '.join(n for n, r in zip(names, res) if not r))
    print('PASS game: ' + ' '.join(names))


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['check']))
