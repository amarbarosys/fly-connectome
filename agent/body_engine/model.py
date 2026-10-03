"""Any MuJoCo model (fly, spider, any MJCF body) to the engine's two flat arrays, plus the oracle that
checks each ported stage against MuJoCo itself.

The engine reads no MJCF. `export` writes mi.bin (int32) and mf.bin (float32): the mjModel fields a
ported stage needs, by their MuJoCo names, concatenated in the order of INTS / FLOATS. Sizes go to
sizes.txt as -D flags, so each body type gets its own compiled engine with its loop bounds and array
offsets as constants (the shiu_worker pattern). A new species is a new MJCF file, not new code.

`check STAGE`: random states (free joint anywhere, any orientation; hinges and slides inside their
ranges), MuJoCo runs the stage in float64, the engine runs it in float32 on the GPU, and every
output field must agree to REL x max(1, |value|).

usage: python -m agent.body_engine.model export MJCF OUT_DIR
       python -m agent.body_engine.model check  MJCF OUT_DIR STAGE [STATES]
"""
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

INTS = ('body_parentid', 'body_jntadr', 'body_jntnum', 'jnt_type', 'jnt_qposadr', 'geom_bodyid')
FLOATS = ('body_pos', 'body_quat', 'jnt_pos', 'jnt_axis', 'qpos0', 'body_ipos', 'body_iquat', 'geom_pos', 'geom_quat')
STAGES = {'kinematics': (mujoco.mj_kinematics, ('xpos', 'xquat', 'xipos', 'geom_xpos', 'xanchor', 'xaxis'))}
REL = 1e-5
ENGINE = Path(__file__).parent


def load(mjcf):
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    if m.nmocap:
        raise SystemExit('FAIL export: mocap bodies are not supported')
    return m


def export(mjcf, out):
    m, out = load(mjcf), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.concatenate([getattr(m, f).ravel() for f in INTS]).astype(np.int32).tofile(out / 'mi.bin')
    np.concatenate([getattr(m, f).ravel() for f in FLOATS]).astype(np.float32).tofile(out / 'mf.bin')
    sizes = {'NBODY': m.nbody, 'NJNT': m.njnt, 'NQ': m.nq, 'NGEOM': m.ngeom}
    (out / 'sizes.txt').write_text(' '.join(f'-D{k}={v}' for k, v in sizes.items()) + '\n')
    print(f'{mjcf}: {sizes} -> {out}')
    return m


def states(m, count, seed=0):
    """(count, nq) qpos: free joints anywhere within 5 mm, random unit quaternion; hinges and slides
    uniform in range (or +-pi when unlimited); ball joints random unit quaternion."""
    rng, q = np.random.default_rng(seed), np.tile(m.qpos0, (count, 1))
    for j in range(m.njnt):
        a, t = m.jnt_qposadr[j], m.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            q[:, a:a + 3] = rng.uniform(-5, 5, (count, 3))
        if t in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            r = rng.normal(size=(count, 4))
            q[:, a + 3 * (t == mujoco.mjtJoint.mjJNT_FREE):][:, :4] = r / np.linalg.norm(r, axis=1, keepdims=True)
        else:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            q[:, a] = rng.uniform(lo, hi, count)
    return q


def check(mjcf, out, stage, count=64):
    m, out = export(mjcf, out), Path(out)
    fn, fields = STAGES[stage]
    q = states(m, count)
    d, want = mujoco.MjData(m), []
    for row in q:
        d.qpos[:] = row
        fn(m, d)
        want.append(np.concatenate([getattr(d, f)[1 if f in ('xpos', 'xquat', 'xipos') else 0:].ravel() for f in fields]))
    want = np.array(want)
    q.astype(np.float32).T.copy().tofile(out / 'qpos.bin')  # (nq, flies): flies fastest, as the engine reads it
    binary = out / f'{stage}_{count}'
    flags = (out / 'sizes.txt').read_text().split()
    subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', *flags, f'-DFLIES={count}', str(ENGINE / f'{stage}.mojo'), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(out) + '/'], check=True)
    got = np.fromfile(out / f'{stage}.bin', np.float32).reshape(-1, count).T
    if got.shape != want.shape:
        raise SystemExit(f'FAIL {stage}: engine wrote {got.shape}, MuJoCo {want.shape}')
    err = np.abs(got - want) / np.maximum(1, np.abs(want))
    bad, start = [], 0
    for f in fields:
        size = getattr(d, f)[1 if f in ('xpos', 'xquat', 'xipos') else 0:].size
        e = float(err[:, start:start + size].max()) if size else 0.
        print(f'{"PASS" if e <= REL else "FAIL"} {f:10s} max rel err {e:.2e}')
        bad += [] if e <= REL else [f]
        start += size
    if bad:
        print(f'FAIL {len(bad)}/{len(fields)} {stage} fields: {" ".join(bad)}')
        sys.exit(1)
    print(f'PASS {stage}: {len(fields)} fields x {count} random states within {REL:g}')


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    export(*args) if cmd == 'export' else check(args[0], args[1], args[2], *map(int, args[3:]))
