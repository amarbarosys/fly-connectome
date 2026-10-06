"""A stone or a wall in the walking fly's path: do the exported convex pairs (ellipsoid-vs-mesh, box-vs-mesh) act in the live
engine (fly_link transport) the way they act in MuJoCo? (fly_link exports them since 75b81d8; the 1 s brain walk meets neither,
so no loop had touched them.) `python -m agent.body_engine.stone_check [stone|wall]`, stone by default.

Scene: body-v2 with stone_0 (ellipsoid 1.8 x 1.35 x 0.3 mm), or wall_0 (box 1 x 42 x 6 mm, turned square to the heading),
moved so its near side is AHEAD mm in front of the settled fly along its heading. The
gait is open loop (fly_link.Gait, fixed symmetric drive), so every arm gets the same ctrl stream:
  engine stone, engine no stone (stone_0 where it was), MuJoCo stone (CPU mj_step, same ctrl), mutant: engine stone, convex off.
Readouts per arm: steps with any fly-stone contact (MuJoCo collision on the arm's own qpos) and the stone's effect on the
path: root xy at the end minus the no-stone arm's (MuJoCo's own no-stone run for MuJoCo).
Every arm starts from the no-stone arm's settled state (settling next to the stone moved the fly 3 mm).
Gates: (1) MuJoCo and the engine both touch the stone >= 20 steps; (2) effect > 0.3 mm in both, engine vs MuJoCo direction
cos >= 0.7 and size ratio 0.5-2; mutant: convex off, effect < 0.05 mm (the stone is passed through). (Peak height was the
first readout: the settled start, 1.6 mm, is the highest point in every arm.)
  gpu-wait run --shared --vram 4 --priority 60 --timeout 2400 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.stone_check
"""
import json
import re
import sys

import mujoco
import numpy as np

from pathlib import Path

from agent.body_engine.fly_link import K, MJCF, OUT, Engine, Gait, prepare

AHEAD, STEPS, DRIVE = 4.0, 800, (0.6, 0.6)
WALL_HALF = 0.5  # wall_0's half thickness, mm


def obstacle_mjcf(kind):
    """body-v2 with the obstacle AHEAD mm in front of the settled fly, written next to the original (its meshes are relative)."""
    d, m, _ = prepare()
    q = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    w, x, y, z = q[3:7]
    h = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    src = Path(MJCF).read_text()
    out = Path(MJCF).parent / f'body-{kind}.xml'
    if kind == 'stone':
        p = q[:2] + AHEAD * np.array([np.cos(h), np.sin(h)])
        out.write_text(re.sub(r'(name="stone_0"[^>]*pos=")[^"]*"', rf'\g<1>{p[0]:.4f} {p[1]:.4f} 0"', src, count=1))
    else:  # the box's thin local x axis along the heading, near face AHEAD mm out
        p = q[:2] + (AHEAD + WALL_HALF) * np.array([np.cos(h), np.sin(h)])
        out.write_text(re.sub(r'(name="wall_0"[^>]*pos=")[^"]*"', rf'\g<1>{p[0]:.4f} {p[1]:.4f} 3" axisangle="0 0 1 {h:.6f}"', src, count=1))
    return out, p


def touches(m, d, q, stone):
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    return any(stone in (d.contact[c].geom1, d.contact[c].geom2) for c in range(d.ncon))


def main(kind):
    xml, p = obstacle_mjcf(kind)
    name = f'{kind}_0'
    arms = {'stone': (xml, True), 'nostone': (MJCF, True), 'mutant': (xml, False)}
    prep = {k: prepare(OUT / f'{kind}-{k}', mjcf=x, convex=c)[:2] for k, (x, c) in arms.items()}
    for k, (d, _) in prep.items():  # one start for every arm: the no-stone settled state (the stone is outside its footprint)
        for n in ('qpos', 'qvel', 'ctrl'):
            (d / f'{n}.bin').write_bytes((prep['nostone'][0] / f'{n}.bin').read_bytes())
    eng = {k: Engine(d, K, f'{kind}-{k}', STEPS) for k, (d, _) in prep.items()}
    d0, m = prep['stone']
    gait = Gait(m, d0, K)
    ctrl = [gait.ctrl(DRIVE) for _ in range(STEPS)]
    tr = {k: [] for k in arms}
    for c in ctrl:
        for k in arms:
            eng[k].send(c)
        for k in arms:
            tr[k].append(eng[k].recv())
    for e in eng.values():
        e.close()
    for k, mm in (('mujoco', m), ('mujoco_nostone', prep['nostone'][1])):  # MuJoCo from the same start state and ctrl stream
        mj = mujoco.MjData(mm)
        mj.qpos[:] = np.fromfile(d0 / 'qpos.bin', np.float32)
        mj.qvel[:] = np.fromfile(d0 / 'qvel.bin', np.float32)
        tr[k] = []
        for c in ctrl:
            mj.ctrl[:] = c
            for _ in range(K):
                mujoco.mj_step(mm, mj)
            tr[k].append(mj.qpos.copy())
    stone = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    probe = mujoco.MjData(m)
    hit = {k: int(sum(touches(m, probe, q, stone) for q in tr[k])) for k in ('stone', 'mujoco', 'mutant')}
    end = {k: np.array(v)[-1, :2] for k, v in tr.items()}
    eff = {'engine': end['stone'] - end['nostone'], 'mujoco': end['mujoco'] - end['mujoco_nostone'], 'mutant': end['mutant'] - end['nostone']}
    size = {k: float(np.linalg.norm(v)) for k, v in eff.items()}
    cos = float(eff['engine'] @ eff['mujoco'] / max(size['engine'] * size['mujoco'], 1e-9))
    ratio = size['engine'] / max(size['mujoco'], 1e-9)
    print(json.dumps({'obstacle': name, 'xy': p.tolist(), 'contact_steps': hit, 'effect_mm': {k: v.round(3).tolist() for k, v in eff.items()},
                      'root_end': {k: np.array(v)[-1, :3].round(3).tolist() for k, v in tr.items()}}))
    ok = [(f"contact: engine {hit['stone']}, MuJoCo {hit['mujoco']} steps >= 20", min(hit['stone'], hit['mujoco']) >= 20),
          (f"effect: engine {size['engine']:.2f}, MuJoCo {size['mujoco']:.2f} mm (> 0.3), cos {cos:.2f} >= 0.7, ratio {ratio:.2f} in 0.5-2",
           min(size['engine'], size['mujoco']) > 0.3 and cos >= 0.7 and 0.5 <= ratio <= 2),
          (f"mutant convex off: effect {size['mutant']:.3f} mm < 0.05", size['mutant'] < 0.05)]
    for k, v in ok:
        print(('PASS ' if v else 'FAIL ') + k)
    if not all(v for _, v in ok):
        sys.exit(f'FAIL {kind} in the path')
    print(f'PASS {kind} in the path: the engine meets it as MuJoCo does')


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'stone'
    if kind not in ('stone', 'wall'): sys.exit(f'usage: stone_check [stone|wall], got {kind!r}')
    main(kind)
