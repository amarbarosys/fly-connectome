"""Brain-driven fly in the live engine, ONE DeviceContext (unit 5b; docs/plans/BRAIN_BODY_LOOP.md step 4): live.mojo -DBRAIN=3.

  python -m agent.body_engine.fly_link3 build
  gpu-wait run --shared --vram 4 --priority 60 --timeout 3000 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.body_engine.fly_link3 check

fly_link.py's loop with the host taken out: the resident Shiu brain (agent/shiu_resident.mojo, the shiu_worker kernels imported), the descending-neuron
readout, the two drives and the FlyGym gait (fly_gait.mojo, parity-gated against HybridTurningController by `check.sh flygait`) all run on live.mojo's DeviceContext,
in the same launch stream as the K-step body launch and the renderer; the host sends nothing and reads nothing per step. Conditions (one live process each, run one
after the other): sym (both eyes 150 Hz; the reference rate of each side is measured on the card at the end of the first 50 control steps and printed), left (left
eye strong), right, silenced (readout zeroed), held (ctrl frozen from step 500, the brain runs on). Gates, as fly_link's: walk > 1 body length with tilt < 20 deg; opposite
yaw for left/right; silenced and held (last half) move < 25 % of sym. Plus: (nohost) memory copies seen by rocprofv3 --memory-copy-trace are the same for a 100-step and a
300-step run with render and packets off, so none per step; cost per frame split (brain + gait / physics / render / copy) from the engine's own phase timers.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from agent.body_engine import fly_gait, fly_link, live, spectate_check as sc
from agent.neural_connectome import ROOT
from agent.shiu import F_POI, V_TH, W_SYN

OUT = Path('.work/agent/body_engine/fly-link3')
K, STEPS, WINDOW = fly_link.K, fly_link.STEPS, fly_link.WINDOW
CONDITIONS = ('sym', 'left', 'right', 'silenced', 'held')


def shiu_flags(n, edges, ndn, ndnl, nseg):
    return [f'-DSHIU_N={n}', f'-DSHIU_E={edges}', '-DSHIU_C=1', f'-DFLY_NDN={ndn}', f'-DFLY_NDNL={ndnl}', f'-DGSEG={nseg}']


def prepare():
    """Engine files of the settled fly, the gait tables, and one brain work dir per condition (stimulus files, DN ids, config) beside them."""
    from agent import shiu_g1
    from agent.experiment import cells
    from agent.shiu import ShiuLive
    d, m, g, nseg = fly_gait.export()
    c = cells()
    data = shiu_g1.flywire()
    graph = next(shiu_g1.arms(data, 'flywire'))[1]
    n = len(data[3])
    eye = {s: np.flatnonzero(np.isin(c.type, fly_link.LOOM) & (c.side == s)) for s in ('left', 'right')}
    dn = {s: np.flatnonzero((c.super == 'descending') & (c.side == s)) for s in ('left', 'right')}
    ids = np.concatenate([dn['left'], dn['right']]).astype(np.int32)
    works = {}
    for cond in CONDITIONS:
        hl, hr = {'left': (fly_link.HZ, fly_link.WEAK), 'right': (fly_link.WEAK, fly_link.HZ)}.get(cond, (fly_link.HZ, fly_link.HZ))
        b = ShiuLive(*graph, n=n, flies=1, seed=71)
        b.stimulate(eye['left'], hl, fly=0, key=0)
        b.stimulate(eye['right'], hr, fly=0, key=0)
        w = OUT / cond
        w.mkdir(parents=True, exist_ok=True)
        entries, p = b._stim_arrays()
        entries.astype(np.int32).tofile(w / 'stim_index.bin')
        b._stim_keys(entries).astype(np.int32).tofile(w / 'stim_key.bin')
        p.tofile(w / 'stim_p.bin')
        b._write_drive(w, entries)
        np.asarray([b.em, b.eg, b.c, V_TH, W_SYN * F_POI], np.float32).tofile(w / 'consts.bin')
        np.asarray([71, b.refractory], np.int32).tofile(w / 'seed.bin')
        ids.tofile(w / 'fly3_dn.bin')
        works[cond] = (w, b)
    return d, m, g, nseg, n, works[CONDITIONS[0]][1], len(ids), len(dn['left']), works


def write_cfg(w, ref, silence=0, freeze=-1):
    np.asarray([ref[0], ref[1], silence, freeze, WINDOW], np.float64).tofile(w / 'fly3.bin')


def binary(d, flags, tag='fly3'):
    return sc.build(d, tag, ['-DPROFILE=1', *flags], K, brain=3, extra_inc=[ROOT / 'agent'])


def run(d, exe, work, graph, steps, render, packets, env_extra=None):
    """One live process; returns (qpos per step if packets, stderr)."""
    import os
    sizes = (d / 'sizes.txt').read_text()
    nu, nq, nv, ns = (int(re.search(rf'-D{k}=(\d+)', sizes).group(1)) for k in ('NU', 'NQ', 'NV', 'NSLOT'))
    size = 4 * (nu + nq + nv + ns)
    env = {**os.environ, 'FLY3_WORK': str(work) + '/', 'FLY3_GRAPH': str(graph) + '/', **(env_extra or {})}
    p = subprocess.Popen(['bash', '-c', 'exec "$0" "$@" 4>/dev/null', str(exe), str(d) + '/', str(steps), *sc.CAM, str(render), str(packets)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    qs = []
    if packets:
        for i in range(steps // packets):
            buf = p.stdout.read(size)
            if len(buf) != size:
                raise SystemExit(f'FAIL live3 ended early at packet {i}: {p.stderr.read().decode()[-400:]}')
            qs.append(np.frombuffer(buf, np.float32)[nu:nu + nq].astype(float))
    p.stdout.read()
    err = p.stderr.read().decode()
    if p.wait():
        raise SystemExit(f'FAIL live3 exited {p.returncode}: {err[-400:]}')
    return np.array(qs), err


def check(probe=False):
    """probe: the same run at 100 control steps (5 % of 1,000, unit-5b rule 14): prints the metrics, asserts nothing but that the loop runs."""
    global STEPS
    STEPS = 100 if probe else STEPS
    OUT.mkdir(parents=True, exist_ok=True)
    d, m, g, nseg, n, brain, ndn, ndnl, works = prepare()
    exe = binary(d, shiu_flags(n, len(brain.col), ndn, ndnl, nseg))
    q0 = np.fromfile(d / 'qpos.bin', np.float32).astype(float)
    root = live.roots(m)[0]
    graph = brain.graph
    tr, ref = {}, None
    for cond in CONDITIONS:
        w = works[cond][0]
        write_cfg(w, ref or (0.0, 0.0), silence=1 if cond == 'silenced' else 0, freeze=STEPS // 2 if cond == 'held' else -1)
        qs, err = run(d, exe, w, graph, STEPS, 0, 1)
        tr[cond] = np.vstack([q0, qs])
        if cond == 'sym':
            r = re.search(r'REF (\S+) (\S+)', err)
            ref = (float(r.group(1)), float(r.group(2)))
        print(f'  {cond}: {STEPS} control steps, {err.strip().splitlines()[-1][:150] if err.strip() else ""}', flush=True)
    res = {c: fly_link.metrics(m, root, tr[c]) for c in CONDITIONS}
    f = {c: float(res[c]['forward'][-1]) for c in CONDITIONS}
    half = STEPS // 2
    last = {c: float(res[c]['forward'][-1] - res[c]['forward'][half]) for c in CONDITIONS}
    yaw = {c: float(res[c]['yaw_deg'][-1]) for c in CONDITIONS}
    bl = fly_link.BODY_LENGTH
    walk = f['sym'] > bl and res['sym']['tilt_max'] < 20
    print(('PASS' if walk else 'FAIL'), f"walk: sym forward {f['sym']:.2f} mm = {f['sym'] / bl:.2f} body lengths in {STEPS * K * 1e-4:g} s (> 1), max tilt {res['sym']['tilt_max']:.1f} deg (< 20)")
    turn = yaw['left'] * yaw['right'] < 0 and min(abs(yaw['left']), abs(yaw['right'])) > 10
    print(('PASS' if turn else 'FAIL'), f"turn: yaw change left-eye-strong {yaw['left']:+.1f} deg, right-eye-strong {yaw['right']:+.1f} deg (opposite signs, each beyond 10)")
    sil = abs(f['silenced']) < 0.25 * f['sym']
    print(('PASS' if sil else 'FAIL'), f"silenced DN: forward {f['silenced']:.2f} mm vs sym {f['sym']:.2f} (< 25 %)")
    held = abs(last['held']) < 0.25 * last['sym']
    print(('PASS' if held else 'FAIL'), f"held ctrl from 0.5 s: last-half forward {last['held']:.2f} mm vs live {last['sym']:.2f} (< 25 %)")
    print(f'reference DN rate per side measured on the card (spikes/s): {ref[0]:.0f} {ref[1]:.0f} (host loop of fly_link: 2840 2680)')
    ok = walk and turn and sil and held
    # no host copy per step: rocprofv3 --memory-copy-trace, 100 vs 300 steps, render and packets off
    copies = {}
    w = works['sym'][0]
    write_cfg(w, ref)
    for steps in (100, 300):
        out = OUT / f'rocprof-{steps}'
        subprocess.run(['/opt/rocm/bin/rocprofv3', '--memory-copy-trace', '--output-format', 'csv', '-d', str(out), '-o', 'mc', '--', 'bash', '-c',
                        f'FLY3_WORK={w}/ FLY3_GRAPH={graph}/ exec "{exe}" "{d}/" {steps} {" ".join(sc.CAM)} 0 0 4>/dev/null >/dev/null 2>/dev/null'],
                       check=True, capture_output=True)
        csv = next(out.rglob('*memory_copy_trace.csv'))
        copies[steps] = len(csv.read_text().strip().splitlines()) - 1
    nohost = copies[100] == copies[300]
    ok &= nohost
    print(('PASS' if nohost else 'FAIL'), f'no host copy per step: memory copies in the whole run {copies[100]} at 100 steps, {copies[300]} at 300 steps (equal: none per step; render and packets off)')
    # cost split, one fly, render each step, 100 steps
    _, err = run(d, exe, w, graph, 100, 1, 0)
    cm = re.search(r'COST physics (\S+) neural_control (\S+) render (\S+) copy (\S+) display (\S+) frames (\d+)', err)
    cost = dict(zip(('physics', 'brain_and_gait', 'render', 'copy', 'display'), map(float, cm.groups()[:5])))
    print('cost ms/frame (one fly, K=10, 100 steps, frames to /dev/null, phases synced): ' + ', '.join(f'{k} {v:.2f}' for k, v in cost.items()))
    (OUT / 'receipt.json').write_text(json.dumps({'forward_mm': f, 'last_half_mm': last, 'yaw_deg': yaw, 'ref': ref, 'copies': copies, 'cost_ms': cost,
                                                  'tilt_max': {c: res[c]['tilt_max'] for c in CONDITIONS}}, indent=1))
    if probe:
        print('PROBE fly_link3 done (gates need the full 1 s)')
        return
    if not ok:
        sys.exit('FAIL fly_link3')
    print('PASS fly_link3: brain-driven fly, one DeviceContext, no host copy per step')


if __name__ == '__main__':
    if (sys.argv[1] if len(sys.argv) > 1 else 'check') == 'build':
        d, m, g, nseg, n, brain, ndn, ndnl, works = prepare()
        print('built', binary(d, shiu_flags(n, len(brain.col), ndn, ndnl, nseg)))
    else:
        check(probe=len(sys.argv) > 2 and sys.argv[2] == 'probe')
