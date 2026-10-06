"""Play the fly by firing its neurons (docs/plans/PLAY.md): keys inject Poisson input into named neurons of the resident
Shiu brain (agent/body_engine/keymap.toml), the brain's descending neurons drive the body through fly_link's readout and
gait, and the brain's own drives can outvote the player.

  gpu-wait run --priority 90 --vram 4 --timeout 3600 -- .venv/bin/python -m agent.body_engine.play          (interactive)
  gpu-wait run --priority 60 --vram 4 --timeout 1200 -- .venv/bin/python -m agent.body_engine.play check

Input: the terminal in kitty keyboard mode (press / release events; ghostty, kitty, foot) and SGR mouse motion; frames go
to mpv. Every key and the mouse are stim entries registered once at start (rate 0); per control step only the probability
row changes, written to the worker's stim_p.bin, which it rereads on every advance (no new kernel, nothing when no key
changed). Walking keys are scaled by the engine's stamina (-DGAME=1 game rows, read from the sensor packet). The world
fires a looming threat (LC4 + LPLC2, both eyes) every few sim seconds or on `t`. HUD (terminal line): the window's DN
spikes split into the player's DNs (key targets) and the brain's own DNs; OVERRIDE when own > player while a key is held.
Engine HUD bars show energy / health / stamina in the frame.

check (headless, one engine F = 5, one brain with 5 columns, 1 s each): idle stands; W walks > 1 body length, tilt < 20;
A and D turn opposite ways beyond 10 deg; W with the threat from 0.5 s: own share rises over player share; stamina under W
below idle; neural cost per step with and without a stim write.
"""
import json
import os
import re
import select
import subprocess
import sys
import termios
import time
import tomllib
import tty
from collections import deque
from pathlib import Path

import numpy as np

from agent.body_engine import fly_link as fl, game_check as gc, live, spectate_check as sc

KEYMAP = Path(__file__).with_name('keymap.toml')
PRESETS = fl.OUT.parents[2] / 'shiu/presets'
OUT = fl.OUT.parent / 'play'
WORK_RATE = float(os.environ.get("PLAY_WORK_RATE", 2e-6))  # stamina + energy per unit of actuator work: W walking does about 7.3e3 per sim second (check 2), so about 70 s to exhaustion; see docs/plans/PLAY.md
NAMES = ('idle', 'w', 'a', 'd', 'w+threat')


def resolve(target, c):
    """Graph indices of a keymap target; a preset name must be a verified preset."""
    from agent import presets
    if target == 'sugar-both':
        return presets.sugar()
    for f in PRESETS.glob('*.json'):
        p = json.loads(f.read_text()).get(target)
        if isinstance(p, dict) and not p.get('verified', True):
            raise SystemExit(f'FAIL keymap target {target}: preset in {f.name} is not verified')
    t, side = (target.rsplit('-', 1) + [''])[:2] if target.endswith(('-left', '-right')) else (target, '')
    ids = np.flatnonzero((c.type == t) & ((c.side == side) if side else True))
    if not len(ids):
        raise SystemExit(f'FAIL keymap target {target}: no cells of that type')
    return ids


class Brain:
    """Resident brain, `cols` columns; every input of keymap.toml a stim entry; readout = DNs per side per column."""
    def __init__(self, cols, km=None):
        from agent import shiu_g1
        from agent.experiment import cells
        from agent.shiu import ShiuLive
        km = km or tomllib.loads(KEYMAP.read_text())
        c = cells()
        data = shiu_g1.flywire()
        self.n, self.cols = len(data[3]), cols
        self.inputs = {k: (np.concatenate([resolve(t, c) for t in v['targets']]), v['hz'], v.get('walk', False)) for k, v in km['keys'].items()}
        ms = km['mouse']
        self.inputs |= {'mouse-' + s: (resolve(ms[s], c), 0, True) for s in ('left', 'right')}
        self.inputs['threat'] = (np.concatenate([resolve(t, c) for t in km['threat']['targets']]), km['threat']['hz'], False)
        self.km, self.names = km, list(self.inputs)
        self.brain = ShiuLive(*next(shiu_g1.arms(data, 'flywire'))[1], n=self.n, flies=cols, seed=71)
        for col in range(cols):
            for ids, _, _ in self.inputs.values():
                self.brain.stimulate(ids, 0, fly=col)
        pos = {e: i for i, e in enumerate(self.brain.stim)}
        self.M = np.zeros((cols, len(self.names), len(pos)), np.float32)  # which entries each input of each column drives
        for col in range(cols):
            for j, (ids, _, _) in enumerate(self.inputs.values()):
                self.M[col, j, [pos[col * self.n + i] for i in ids]] = 1
        self.hz = np.array([v[1] for v in self.inputs.values()], np.float32)
        self.walk = np.array([v[2] for v in self.inputs.values()])
        dn = {s: np.flatnonzero((c.super == 'descending') & (c.side == s)) for s in ('left', 'right')}
        player = np.unique(np.concatenate([v[0] for k, v in self.inputs.items() if k != 'threat']))
        self.readout = np.concatenate([col * self.n + dn[s] for col in range(cols) for s in ('left', 'right')]).astype(np.int32)
        self.side = np.concatenate([np.full(len(dn[s]), si) for col in range(cols) for si, s in enumerate(('left', 'right'))])
        self.colof = np.concatenate([np.full(len(dn['left']) + len(dn['right']), col) for col in range(cols)])
        self.mine = np.isin(self.readout % self.n, player)  # DNs the player stimulates directly
        self.brain.start([1], self.readout)
        self.p = None

    def set(self, rates):
        """rates (cols, inputs) in Hz -> the worker's per-entry probabilities, written only when they changed."""
        p = np.clip(np.einsum('cn,cne->e', rates, self.M) * fl.DT, 0, 1).astype(np.float32)
        if self.p is None or not np.array_equal(p, self.p):
            p.tofile(self.brain.work / 'stim_p.bin')
            self.p = p

    def advance(self, k):
        """Cumulative spikes: (cols, 2 sides) all DNs, (cols, 2) player DNs / own DNs."""
        cum = self.brain.advance(k).astype(np.int64)
        side = np.zeros((self.cols, 2), np.int64)
        np.add.at(side, (self.colof, self.side), cum)
        split = np.zeros((self.cols, 2), np.int64)
        np.add.at(split, (self.colof, (~self.mine).astype(int)), cum)
        return side, split


def setup(F, render=0, out='/dev/null'):
    d, m, _ = fl.prepare(OUT / 'fly')
    if not (d / 'mf.base').exists():
        (d / 'mf.bin').rename(d / 'mf.base')
    gc.set_rules(d, 0.0, WORK_RATE)
    eng = fl.Engine(d, fl.K, 'play', 0, F=F, flags=('-DGAME=1', '-DHUD=1', '-DFOLLOW=1'), out=out, render=render)
    return d, m, eng


def loop(brain, eng, gaits, rates_at, steps, on_step=None):
    """Brain + body for `steps` control steps (0 = until rates_at returns None). The first fl.WINDOW steps hold the body and
    fire `w` in every column to fix the drive scale (W = full drive, as fly_link's symmetric eye drive); then rates_at(i, game)."""
    j = brain.names.index('w')
    zero = np.zeros((brain.cols, 2), np.int64)
    cum, split, held = deque([zero], fl.WINDOW + 1), deque([zero], fl.WINDOW + 1), gaits[0].held  # one window of history, fixed size
    ref, tr, cost = None, [], {'neural': 0.0, 'write': 0.0}
    game = np.ones((brain.cols, 4), np.float32)
    i = 0
    while steps == 0 or i < steps + fl.WINDOW:
        if i < fl.WINDOW:
            r = np.zeros((brain.cols, len(brain.names)), np.float32)
            r[:, j] = brain.hz[j]
        else:
            r = rates_at(i - fl.WINDOW, game)
            if r is None:
                break
        t = time.perf_counter()
        brain.set(r)
        cost['write'] += time.perf_counter() - t
        t = time.perf_counter()
        s, sp = brain.advance(fl.K)
        cost['neural'] += time.perf_counter() - t
        cum.append(s)
        split.append(sp)
        w = min(i + 1, fl.WINDOW)
        rate = (cum[-1] - cum[-1 - w]) / (w * fl.K * fl.DT)
        if i + 1 == fl.WINDOW:
            ref = np.maximum(rate.mean(0), 1.0)
        drive = np.clip(rate / ref, 0, 1) if i >= fl.WINDOW else np.zeros_like(rate, float)
        ctrl = np.stack([g.ctrl(dv) for g, dv in zip(gaits, drive)]) if i >= fl.WINDOW else np.tile(held, (brain.cols, 1))
        eng.send(ctrl)
        q = eng.recv()
        game = eng.game
        if i >= fl.WINDOW:
            tr.append(q if brain.cols > 1 else q[None])
            win = split[-1] - split[0]
            if on_step:
                on_step(i - fl.WINDOW, win, game, drive)
        i += 1
    return np.array(tr), ref, cost


def check():
    km = tomllib.loads(KEYMAP.read_text())
    brain = Brain(len(NAMES), km)
    d, m, eng = setup(len(NAMES))
    gaits = [fl.Gait(m, d, fl.K) for _ in NAMES]
    ix = {k: brain.names.index(k) for k in ('w', 'a', 'd', 'threat')}
    half, wins = fl.STEPS // 2, []

    def rates_at(i, game):
        r = np.zeros((len(NAMES), len(brain.names)), np.float32)
        for col, keys in ((1, 'w'), (2, 'a'), (3, 'd'), (4, 'w')):
            r[col, ix[keys]] = brain.hz[ix[keys]]
        if i >= half:
            r[4, ix['threat']] = brain.hz[ix['threat']]
        return r

    tr, ref, cost = loop(brain, eng, gaits, rates_at, fl.STEPS, lambda i, win, game, drive: wins.append(win.copy()))
    brain.brain.close()
    engine_cost = eng.close()
    root = live.roots(m)[0]
    res = {n: fl.metrics(m, root, np.r_[tr[:1, col], tr[:, col]]) for col, n in enumerate(NAMES)}
    game = eng.game
    wins = np.array(wins, float)  # (steps, cols, [player, own])
    before = wins[half - fl.WINDOW:half, 4].sum(0)
    during = wins[-fl.WINDOW:, 4].sum(0)
    fwd = {n: float(res[n]['forward'][-1]) / fl.BODY_LENGTH for n in NAMES}
    yaw = {n: float(res[n]['yaw_deg'][-1]) for n in NAMES}
    n = fl.STEPS + fl.WINDOW
    gates = {
        'idle stands': abs(fwd['idle']) < 0.25,
        'w walks': fwd['w'] > 1 and res['w']['tilt_max'] < 20,
        'a/d turn': yaw['a'] * yaw['d'] < 0 and min(abs(yaw['a']), abs(yaw['d'])) > 10,
        'threat overrides': during[1] > during[0] and during[1] / max(during.sum(), 1) > before[1] / max(before.sum(), 1),
        'stamina drains under w': game[1, 2] < game[0, 2],
    }
    out = {'forward_body_lengths': fwd, 'yaw_deg': yaw, 'tilt_max': {k: res[k]['tilt_max'] for k in NAMES},
           'w+threat [player, own] DN spikes, last window before / during': [before.tolist(), during.tolist()],
           'game rows end (energy, health, stamina, dead)': {k: game[c].tolist() for c, k in enumerate(NAMES)},
           'drive ref (DN Hz, W held)': ref.tolist(), 'ms per control step': {k: 1e3 * v / n for k, v in cost.items()},
           'engine COST ms/frame': engine_cost, 'work_rate': WORK_RATE, 'gates': gates}
    (OUT / 'check.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for k, v in out.items():
        print(k, json.dumps(v, default=float))
    print(f'A turns {"left" if yaw["a"] > 0 else "right"} (yaw {yaw["a"]:+.1f} deg), D {yaw["d"]:+.1f} deg')
    bad = [k for k, v in gates.items() if not v]
    print('\n'.join(f"{'FAIL' if k in bad else 'PASS'} {k}" for k in gates))
    if bad:
        sys.exit(f'FAIL {len(bad)}/{len(gates)} {bad}')


KEY = re.compile(rb'\x1b\[(\d+)(?::\d+)*(?:;\d+(?::(\d))?)?u|\x1b\[<(\d+);(\d+);(\d+)[Mm]|([ -~])')


def play():
    km = tomllib.loads(KEYMAP.read_text())
    brain = Brain(1, km)
    fifo = OUT / 'frames.fifo'
    OUT.mkdir(parents=True, exist_ok=True)
    fifo.unlink(missing_ok=True)
    os.mkfifo(fifo)
    mpv = subprocess.Popen(['mpv', '--demuxer=rawvideo', f'--demuxer-rawvideo-w={sc.W}', f'--demuxer-rawvideo-h={sc.H}',
                            '--demuxer-rawvideo-mp-format=rgba', '--profile=low-latency', '--untimed', '--title=fly', str(fifo)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d, m, eng = setup(1, render=4, out=fifo)
    gaits = [fl.Gait(m, d, fl.K)]
    th = km['threat']
    every, dur = round(th['every_s'] / (fl.K * fl.DT)), round(th['dur_s'] / (fl.K * fl.DT))
    names = {'space': ' '} | {k: k for k in km['keys']}
    held, tap, mouse, lastx, threat_until = set(), {}, 0.0, None, -1
    fd, old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin.fileno())
    tty.setraw(fd)
    sys.stdout.write('\x1b[>11u\x1b[?1003h\x1b[?1006h')  # kitty keys with press/release, SGR mouse motion
    sys.stdout.flush()

    def rates_at(i, game):
        nonlocal mouse, lastx, threat_until
        while select.select([fd], [], [], 0)[0]:
            for g in KEY.finditer(os.read(fd, 4096)):
                if g.group(1):
                    ch, ev = chr(int(g.group(1))), g.group(2) or b'1'
                    if ch in 'q\x1b':
                        return None
                    (held.discard if ev == b'3' else held.add)(ch)
                elif g.group(3):
                    x = int(g.group(4))
                    mouse += 0 if lastx is None else x - lastx
                    lastx = x
                elif g.group(6):  # a terminal without kitty events: a keypress holds for 0.3 s
                    ch = g.group(6).decode()
                    if ch == 'q':
                        return None
                    tap[ch] = i + round(0.3 / (fl.K * fl.DT))
        keys = held | {k for k, t in tap.items() if t > i}
        if th['key'] in keys or i % every == 0:
            threat_until = i + dur
        r = np.zeros((1, len(brain.names)), np.float32)
        for k, ch in names.items():
            r[0, brain.names.index(k)] = brain.hz[brain.names.index(k)] * (ch in keys)
        mouse *= 0.9
        hz = min(abs(mouse) * km['mouse']['hz_per_px'], km['mouse']['max_hz'])
        r[0, brain.names.index('mouse-left' if mouse < 0 else 'mouse-right')] = hz
        r[0, brain.walk] *= 0.2 + 0.8 * game[0, 2]  # tired: the player's walking drive weakens, the brain's own does not
        r[0, brain.names.index('threat')] = brain.hz[brain.names.index('threat')] * (i < threat_until)
        return r

    def hud(i, win, game, drive):
        if i % 20:
            return
        p, o = win[0]
        keys = held | {k for k, t in tap.items() if t > i}
        flag = ' OVERRIDE' if keys and o > p else ''
        sys.stdout.write(f'\r\x1b[2Kplayer {p:5d} own {o:5d} | drive L {drive[0, 0]:.2f} R {drive[0, 1]:.2f} | E {game[0, 0]:.2f} '
                         f'S {game[0, 2]:.2f}{" DEAD" if game[0, 3] else ""} | {"".join(sorted(keys))}{" THREAT" if i < threat_until else ""}{flag}')
        sys.stdout.flush()

    try:
        loop(brain, eng, gaits, rates_at, 0, hud)
    finally:
        sys.stdout.write('\x1b[<u\x1b[?1003l\x1b[?1006l\r\n')
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        brain.brain.close()
        eng.close()
        mpv.terminate()


if __name__ == '__main__':
    {'check': check, 'play': play}[sys.argv[1] if len(sys.argv) > 1 else 'play']()
