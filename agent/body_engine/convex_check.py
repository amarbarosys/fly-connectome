"""Narrowphase parity gate of convex.mojo (GJK + EPA + multicontact) against MuJoCo 3.9 mj_collision on the fly MJCF.

Cases: a pair's two geoms at random relative poses (centres within a ball of their summed bounding radii, so a mix of
separated, shallow, touching and deep), MuJoCo's contact for exactly that pair as the oracle. Compared: contact
count, dist, pos, normal. The CPU driver convex_test.mojo is built float64 (parity with MuJoCo's doubles, bar 1e-9)
or float32 (the engine; bar measured, stated by the caller).

usage: python -m agent.body_engine.convex_check KIND N [f64|f32] [BAR] [SEED]   KIND: ellipsoid | box
"""
import subprocess, sys, os
import numpy as np
import mujoco
from pathlib import Path

BODY = '.work/agent/gpu-body-port/body-v2/body.xml'
CW, OW = 40, 36
G = mujoco.mjtGeom
# float32 bars (measured: `floor` = MuJoCo f64 vs MuJoCo on float32-rounded poses gives dist 2e-7, pos 3e-4, normal 2e-3 max)
F64 = dict(frac=0.002, max=2e-5)  # fraction of values above the bar, and the worst (EPA tolerance 1e-6 scaled by the lever arm)
F32 = dict(dist_max=3e-4, pos_bar=1e-3, normal_bar=4e-3, frac=0.005, shallow=1e-4, count_frac=0.02)


def hull_tables(m, meshes):
    """Hull vertices (geom frame), first edges (absolute into edges), edge lists, per-hull offsets: model.py feet()'s layout."""
    verts, first, edges, meta, index = [], [], [], [], {}
    for mesh in meshes:
        ga = m.mesh_graphadr[mesh]
        nv = m.mesh_graph[ga]
        edgeadr = m.mesh_graph[ga + 2:ga + 2 + nv]
        gid = m.mesh_graph[ga + 2 + nv:ga + 2 + 2 * nv]
        local = m.mesh_graph[ga + 2 + 2 * nv:]
        local = local[:edgeadr.max() + list(local[edgeadr.max():]).index(-1) + 1]
        index[mesh] = len(meta)
        meta.append((len(verts) // 3, len(first), len(edges)))
        first += list(len(edges) + edgeadr)
        edges += list(local)
        verts += list(m.mesh_vert[m.mesh_vertadr[mesh] + gid].ravel())
    return (np.array(verts, np.float32), np.array(first, np.int32), np.array(edges, np.int32),
            np.array(meta, np.int32), index)


def randrot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat


def run(kind, n, prec='f64', bar=1e-9, seed=0, mutant=None, verbose=True):
    m = mujoco.MjModel.from_xml_path(BODY)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    want = G.mjGEOM_BOX if kind == 'box' else G.mjGEOM_ELLIPSOID
    sel = [i for i in range(m.npair) if m.geom_type[m.pair_geom1[i]] == want and m.geom_type[m.pair_geom2[i]] == G.mjGEOM_MESH]
    meshes = sorted({int(m.geom_dataid[m.pair_geom2[i]]) for i in sel})
    hv, hf, he, hm, hidx = hull_tables(m, meshes)
    rng = np.random.default_rng(seed)
    work = Path('.work/agent/body_engine/convex') / f'{kind}-{prec}'
    work.mkdir(parents=True, exist_ok=True)
    cases = np.zeros((n, CW))
    pairs = []
    for c in range(n):
        i = sel[rng.integers(len(sel))]
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        pairs.append((i, g1, g2))
        r = m.geom_rbound[g1] + m.geom_rbound[g2]
        dirn = rng.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        pos2 = d.geom_xpos[g2].copy()
        mat1, mat2 = randrot(rng), randrot(rng)
        if kind == 'box':  # the mesh centre inside the box grown by the mesh's radius (a box dwarfs a ball of radius r)
            ext = m.geom_size[g1] + 1.2 * m.geom_rbound[g2]
            pos1 = pos2 - mat1.reshape(3, 3) @ rng.uniform(-ext, ext)
        else:
            pos1 = pos2 + dirn * r * rng.uniform(0.0, 0.9)
        row = cases[c]
        row[0] = 0 if kind == 'ellipsoid' else 2
        row[1] = 1
        row[2:5] = m.geom_size[g1]
        row[5:8] = pos1
        row[8:17] = mat1
        row[17:20] = pos2
        row[20:29] = mat2
        row[29] = hidx[int(m.geom_dataid[g2])]
        row[30] = m.pair_margin[i] + m.pair_gap[i]
        row[31] = m.opt.ccd_tolerance
        row[32] = min(m.geom_rbound[g1], m.geom_rbound[g2])
    cases.tofile(work / 'cases.f64')
    np.save(work / 'pairs.npy', np.array(pairs))
    hv.tofile(work / 'hverts.f32'); hf.tofile(work / 'hfirst.i32'); he.tofile(work / 'hedges.i32'); hm.tofile(work / 'hmeta.i32')
    exe = work / 'ct'
    flags = ['-DCCD64=1'] if prec == 'f64' else []
    p = subprocess.run(['/home/<user>/.local/bin/mojo', 'build', *flags, 'agent/body_engine/convex_test.mojo', '-o', str(exe)], capture_output=True, text=True)
    if p.returncode:
        print('FAIL build convex_test:', p.stderr[-2000:])
        return False
    p = subprocess.run([str(exe), str(work) + '/'], capture_output=True, text=True)
    if p.returncode:
        print('FAIL run convex_test:', p.stdout[-500:], p.stderr[-500:])
        return False
    out = np.fromfile(work / 'out.f64').reshape(n, OW)
    bad = {'ncon': 0, 'dist': 0, 'pos': 0, 'normal': 0}
    worst = {'dist': 0.0, 'pos': 0.0, 'normal': 0.0}
    contact_cases = 0
    errs = []
    counts = {}
    first_bad = []
    for c in range(n):
        i, g1, g2 = pairs[c]
        cs = oracle(m, d, g1, g2, cases[c])
        contact_cases += bool(cs)
        mine = int(out[c, 0])
        counts[(len(cs), mine)] = counts.get((len(cs), mine), 0) + 1
        if len(cs) != mine:
            bad['ncon'] += 1
            if len(first_bad) < 5:
                first_bad.append((c, 'ncon', len(cs), mine))
            continue
        for k, (cd, cp, cn) in enumerate(cs):
            ed = abs(out[c, 1 + 7 * k] - cd)
            ep = np.abs(out[c, 2 + 7 * k:5 + 7 * k] - cp).max()
            en = np.abs(out[c, 5 + 7 * k:8 + 7 * k] - cn).max()
            errs.append((ed, ep, en, cases[c][30] - cd))
            worst['dist'], worst['pos'], worst['normal'] = max(worst['dist'], ed), max(worst['pos'], ep), max(worst['normal'], en)
            for key, e in (('dist', ed), ('pos', ep), ('normal', en)):
                if e > bar:
                    bad[key] += 1
                    if len(first_bad) < 5:
                        first_bad.append((c, key, k, e))
    deep = np.array([e[:3] for e in errs if e[3] > F32['shallow']]) if errs else np.zeros((1, 3))  # force-significant contacts
    errs = [e[:3] for e in errs]
    if prec == 'f64':  # exact except path flips: a comparison decided by 1e-16 changes where EPA stops, within its 1e-6 tolerance
        a = np.array(errs) if errs else np.zeros((1, 3))
        over = float((a > bar).mean())
        bad = {'ncon': bad['ncon'], 'dist': int(over > F64['frac'] or a[:, 0].max() > F64['max']),
               'pos': int(a[:, 1].max() > F64['max']), 'normal': int(a[:, 2].max() > F64['max'])}
        print(f'  f64: {100 * over:.3f} % of dist/pos/normal values above {bar:g} (bar {100 * F64["frac"]:.1f} %), max {a.max():.1e} (bar {F64["max"]:g})')
    counts_bad = bad['ncon']
    if prec == 'f32':  # per-quantity bars instead of one bar on the worst case
        a = deep
        p99 = np.percentile(a, 99, axis=0)
        bad = {'ncon': int(bad['ncon'] > F32['count_frac'] * n), 'dist': int(np.array(errs)[:, 0].max() > F32['dist_max']),
               'pos': int((a[:, 1] > F32['pos_bar']).mean() > F32['frac']), 'normal': int((a[:, 2] > F32['normal_bar']).mean() > F32['frac'])}
        print(f"  f32 (contacts deeper than {F32['shallow']:g} inside the margin: {len(a)} of {len(errs)}): p99 pos {p99[1]:.1e} normal {p99[2]:.1e}, max pos {a[:, 1].max():.1e} normal {a[:, 2].max():.1e}; "
              f"above the bars (pos {F32['pos_bar']:g}, normal {F32['normal_bar']:g}): {100 * (a[:, 1] > F32['pos_bar']).mean():.3f} % / {100 * (a[:, 2] > F32['normal_bar']).mean():.3f} % (bar {100 * F32['frac']:.1f} %); "
              f"count mismatches {counts_bad} of {n} cases (bar {100 * F32['count_frac']:.0f} %)")
    ok = sum(bad.values()) == 0 and contact_cases > 0
    if verbose:
        print(f"{'PASS' if ok else 'FAIL'} convex {kind} {prec}: {n} cases, {contact_cases} with a contact, bar {bar:g}; "
              f"worst dist {worst['dist']:.2e} pos {worst['pos']:.2e} normal {worst['normal']:.2e}; mismatches {bad}; contacts oracle/engine {dict(sorted(counts.items()))}")
        if errs:
            q = np.percentile(np.array(errs), [50, 99], axis=0)
            print(f'  p50 / p99 of dist, pos, normal: {q[0][0]:.1e} {q[0][1]:.1e} {q[0][2]:.1e} / {q[1][0]:.1e} {q[1][1]:.1e} {q[1][2]:.1e}')
        if first_bad and not ok:
            print('  first:', first_bad)
    return ok


def oracle(m, d, g1, g2, row):
    """MuJoCo's contact for the pair at the case's poses: (dist, pos, normal) or None."""
    d.geom_xpos[g1] = row[5:8]; d.geom_xmat[g1] = row[8:17]
    d.geom_xpos[g2] = row[17:20]; d.geom_xmat[g2] = row[20:29]
    mujoco.mj_collision(m, d)
    cs = [k for k in range(d.ncon) if d.contact[k].geom1 == g1 and d.contact[k].geom2 == g2]
    return [(d.contact[k].dist, d.contact[k].pos.copy(), d.contact[k].frame[:3].copy()) for k in cs]


def floor(kind, n):
    """The float32 floor of the algorithm itself: MuJoCo (double) on the cases of the last run() against MuJoCo on the same
    cases with every pose rounded to float32 (what any float32 port receives). Prints the worst dist, pos, normal difference."""
    m = mujoco.MjModel.from_xml_path(BODY)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    work = Path('.work/agent/body_engine/convex') / f'{kind}-f64'
    cases = np.fromfile(work / 'cases.f64').reshape(-1, CW)[:n]
    pairs = np.load(work / 'pairs.npy')
    worst, flips = np.zeros(3), 0
    for row, (i, g1, g2) in zip(cases, pairs):
        a = oracle(m, d, g1, g2, row)
        r32 = row.copy()
        r32[2:29] = row[2:29].astype(np.float32)
        b = oracle(m, d, g1, g2, r32)
        if len(a) != len(b):
            flips += 1
        else:
            for x, y in zip(a, b):
                worst = np.maximum(worst, [abs(x[0] - y[0]), np.abs(x[1] - y[1]).max(), np.abs(x[2] - y[2]).max()])
    print(f'floor {kind}: MuJoCo f64 vs MuJoCo on float32-rounded poses, {len(cases)} cases: count flips {flips}, '
          f'worst dist {worst[0]:.2e} pos {worst[1]:.2e} normal {worst[2]:.2e}')


if __name__ == '__main__':
    if sys.argv[1] == 'floor':
        floor(sys.argv[2], int(sys.argv[3]))
        sys.exit(0)
    kind = sys.argv[1]
    n = int(sys.argv[2])
    prec = sys.argv[3] if len(sys.argv) > 3 else 'f64'
    bar = float(sys.argv[4]) if len(sys.argv) > 4 else (1e-9 if prec == 'f64' else 1e-4)
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    sys.exit(0 if run(kind, n, prec, bar, seed) else 1)
