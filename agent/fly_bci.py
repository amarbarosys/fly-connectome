"""A brain-computer interface on the Shiu fly brain: frequency tagging (A), the brain's spectrum (C), and the fly's
descending neurons steering another body, the jumping spider (B). Modelled on agent/visual_side.py.

A  One eye's looming projection neurons (LC4 + LPLC2, left side, 150 Hz Poisson) flicker as a square wave at 8, 13 or
   21 Hz (ShiuBrain.flicker), or stay on (0 Hz control). Spike counts in 5 ms bins for 2 s, per arm (real wiring and
   one block shuffle), for two "electrodes": all descending neurons (DN) and the whole brain (EEG-like). Per trial:
   power spectrum of 0.2-2 s; tagged frequency = argmax power over {8, 13, 21}; SNR = power at f / mean of 4
   neighbouring bins (2-3 bins away). Accuracy over trials; the 0 Hz control must show no tag (SNR ~1).
C  The constant-drive trials: brain-wide spectrum peak frequency and peak/median power (flat = no rhythm).
B  Looming left vs right (constant 150 Hz, 8 trials each): fixed readout, no fitting: steer = (left - right) spikes
   of the looming-escape DN pairs DNp02, DNp04, DNp05, DNp11 in 0-100 ms (visual-side probe 2026-10-03: left loom
   drives the left member up). Escape turns away: turn = -0.8 sign(steer). Each trial's command drives one spider
   (spider_link, card gait kernel, 1 s); gate: spider yaw sign away from the looming side in >= 14/16 trials on the
   real wiring. Model predictions, not fly measurements.

usage: gpu-wait run --priority 80 --timeout 2400 -- .venv/bin/python -m agent.fly_bci  -> .work/agent/shiu/fly-bci.json
"""
import json

import numpy as np

from agent import shiu_g1
from agent.experiment import cells
from agent.neural_connectome import ROOT
from agent.shiu import ShiuGPU

HZ, FREQS, R = 150., (8., 13., 21.), 6
BIN, T = 50, 20000  # 5 ms bins (dt 1e-4), 2 s
ESCAPE = ('DNp02', 'DNp04', 'DNp05', 'DNp11')


def two_arms():
    data = shiu_g1.flywire()
    arms = shiu_g1.arms(data, 'flywire')
    real, block = next(arms), next(a for a in arms if a[0].startswith('block'))
    n = len(data[3])
    pre, post, w = (np.concatenate([np.asarray(g[1][i]) + (t * n if i < 2 else 0) for t, g in enumerate((real, block))]) for i in range(3))
    return (real[0], block[0]), (pre, post, w), n


def spectrum(counts):
    """counts (bins,) -> (freqs, power) of 0.2-2 s, mean removed."""
    x = counts[int(.2 / (BIN * 1e-4)):].astype(float)
    return np.fft.rfftfreq(len(x), BIN * 1e-4), np.abs(np.fft.rfft(x - x.mean())) ** 2


def at(f, hz):
    return int(np.argmin(np.abs(f - hz)))


def snr(f, p, hz):
    k = at(f, hz)
    return p[k] / (np.r_[p[k - 3:k - 1], p[k + 2:k + 4]].mean() + 1e-12)


def part_a_c(names, graph, n, c):
    loom = np.flatnonzero(np.isin(c.type, ('LC4', 'LPLC2')) & (c.side == 'left'))
    dn = np.flatnonzero(c.super == 'descending')
    conds = list(FREQS) + [0.]
    cols = len(conds) * R
    brain = ShiuGPU(*graph, n=2 * n, flies=cols, seed=53)
    for i, hz in enumerate(conds):
        for r in range(R):
            ids = np.concatenate([loom + a * n for a in range(2)])
            if hz:
                brain.flicker(ids, HZ, hz, fly=i * R + r, key=i * R + r)
            else:
                brain.stimulate(ids, HZ, fly=i * R + r, key=i * R + r)
    cgrp = np.full(2 * n, -1, np.int32)
    for a in range(2):
        cgrp[a * n:(a + 1) * n] = 2 * a + 1  # rest of the brain
        cgrp[dn + a * n] = 2 * a  # DNs
    sums, _ = brain.run_groups(list(range(BIN, T + 1, BIN)), cgrp)
    s = np.asarray(sums, np.int64).reshape(T // BIN, 4, cols)
    rate = np.diff(np.concatenate([np.zeros((1, 4, cols), np.int64), s]), axis=0)  # spikes per bin
    out = {}
    for a, name in enumerate(names):
        out[name] = {}
        for el, x in {'DN': rate[:, 2 * a], 'brain': rate[:, 2 * a] + rate[:, 2 * a + 1]}.items():
            res, hit, table = {'spikes_per_s': float(x.sum(0).mean() / 2)}, 0, {}
            for i, hz in enumerate(conds):
                rows = []
                for r in range(R):
                    f, p = spectrum(x[:, i * R + r])
                    rows.append([snr(f, p, g) for g in FREQS])
                    if hz:
                        hit += int(FREQS[int(np.argmax([p[at(f, g)] for g in FREQS]))] == hz)
                    elif r == 0 and el == 'brain':
                        band = (f >= 2) & (f <= 95)
                        k = int(np.argmax(p[band]))
                        res['C_constant_drive'] = {'peak_hz': round(float(f[band][k]), 2),
                                                   'peak_over_median': round(float(p[band][k] / (np.median(p[band]) + 1e-12)), 1)}
                table[f'{hz:g}Hz'] = np.round(np.mean(rows, 0), 1).tolist()
            res['A_tag_accuracy'] = hit / (len(FREQS) * R)
            res['A_snr_at_8_13_21'] = table
            out[name][el] = res
    return out


def part_b(names, graph, n, c):
    from agent.body_engine.spider_link import files, run, yaw_of
    loom = {sd: np.flatnonzero(np.isin(c.type, ('LC4', 'LPLC2')) & (c.side == sd)) for sd in ('left', 'right')}
    esc = {sd: np.flatnonzero(np.isin(c.type, ESCAPE) & (c.side == sd)) for sd in ('left', 'right')}
    K = 8
    brain = ShiuGPU(*graph, n=2 * n, flies=2 * K, seed=59)
    for j, sd in enumerate(('left', 'right')):
        for r in range(K):
            brain.stimulate(np.concatenate([loom[sd] + a * n for a in range(2)]), HZ, fly=j * K + r, key=j * K + r)
    counts = brain.run([1000])[0]  # (2n, 16): 0-100 ms
    want = np.repeat([-1, 1], K)  # left loom -> turn right (yaw < 0)
    out = {}
    for a, name in enumerate(names):
        steer = counts[esc['left'] + a * n].sum(0) - counts[esc['right'] + a * n].sum(0)
        turn = -0.8 * np.sign(steer)
        d = files(2 * K, f'bci-{name}')
        np.r_[np.ones(2 * K), turn].astype(np.float32).tofile(d / 'drive.bin')
        pk, _ = run(d, 2 * K, 10, 2, 1000)
        q = pk[-1][1]
        yaw = np.array([yaw_of(q[:, i]) for i in range(2 * K)])
        out[name] = {'steer': steer.astype(int).tolist(), 'turn': turn.tolist(), 'spider_yaw_deg': np.round(yaw, 0).tolist(),
                     'correct': int((np.sign(yaw) == want).sum()), 'of': 2 * K}
    return out


def main():
    c = cells()
    names, graph, n = two_arms()
    res = {'A_C': part_a_c(names, graph, n, c), 'B': part_b(names, graph, n, c)}
    (ROOT / '.work/agent/shiu/fly-bci.json').write_text(json.dumps(res, indent=2) + '\n')
    for name in names:
        for el, r in res['A_C'][name].items():
            print(f"A {name:12s} {el:5s} {r['spikes_per_s']:9.0f} spk/s  tag acc {r['A_tag_accuracy']:.2f}  SNR {r['A_snr_at_8_13_21']}"
                  + (f"  C {r['C_constant_drive']}" if 'C_constant_drive' in r else ''))
        b = res['B'][name]
        print(f"B {name:12s} spider turned away from the loom {b['correct']}/{b['of']}  yaw {b['spider_yaw_deg']}")
    ok = res['B'][names[0]]['correct'] >= 14 and res['A_C'][names[0]]['DN']['A_tag_accuracy'] >= .9
    print(('PASS' if ok else 'FAIL'), 'fly BCI: real wiring tags flicker in DNs (>= 0.9) and steers the spider away (>= 14/16)')


if __name__ == '__main__':
    main()
