"""Spider drives and reflexes on top of the closed eye loop (private note SPIDER_BRAIN U2c): hunger, thirst, fear,
instinct (an orienting reflex from the secondary eyes) and water seeking. The brain (the trained scaffold pathway) still
drives the principal eyes; the drives below are FIXED rules (not learned), each reading only the rendered eyes and the
animal's own need levels, never a target's coordinates. One rule wins per step, in this order:

  fear     a blob in any eye whose area grew x FEAR_GROW over 50 ms (looming) -> run: turn away, full speed, 0.5 s latched
  hunt     hunger >= 0.2 and hunger >= thirst: prey (pixels < 15) in a principal eye -> orient (turn = gaze / 2 deg, the
           eyes lead); prey only in a secondary eye -> instinct: turn toward its bearing until the principal eyes have it
  drink    thirst >= 0.5: water (pixels > 200) in any eye -> turn toward it and walk; none seen -> turn in place to scan
  idle     stand
Needs: hunger and thirst rise by HUNGER_S / THIRST_S per second; thirst resets to 0 when the head is within WATER_R of the
drop (drinking), hunger when the prey is within CATCH_MM.

check (each scenario 32 spiders, its arms in separate engine runs):
  N  needs (CPU): thirst drains and refills at the water, hunger drains; mutant refill-sign FAILs
  I  instinct: prey still at 22-30 deg in a secondary eye, hungry -> within the principal eyes (< 3 deg) in >= 80 % after 3 s;
     no-reflex arm and the reflex-sign mutant <= 20 %
  H  hunger: prey crossing 6-9 deg/s in the principal eyes: hungry (1.0) body yaw >= 50 % of the bearing change;
     sated (0.1) yaw < 10 % of it and moves < 1 mm
  F  fear: an object approaching at 100 mm/s from 60 mm at 20 deg (0.4 s, it stops 20 mm short): |bearing to it| grows
     >= 15 deg (the spider turned away); the same object receding < 5 deg; mutant (escape turn toward) < 0
  (instinct prey at 20 mm: at 191 mm it is under one pixel in a 70 deg secondary eye)
  W  water: thirsty, sated, the drop at 30 mm / 50 deg left -> within WATER_R in >= 70 % after 8 s and thirst reset;
     not thirsty <= 10 %; mutant (turn away from the water) <= 10 %
usage: gpu-wait run --shared --vram 4 --priority 60 --timeout 5400 --env PYTHONPATH=$PWD -- .venv/bin/python -m agent.spider_drives check
"""
import json
import sys

import numpy as np

from agent import spider_eye_loop as L
from kinestride.models.spider import WATER_R, WATER_XY
from agent.body_engine.spider_gait import gait
from agent.spider_body_loop import wrap, yaw

E, W, H, B, DT = L.E, L.W, L.H, L.G.B, L.P.DT
HUNGER_S, THIRST_S, CATCH_MM, FEAR_GROW, FEAR_STEPS = 0.002, 0.01, 3.0, 1.5, 50
TABLE = E.eye_table(W)
AM = (0, 1)


def blobs(ret):
    """per animal and eye: (prey bearing deg or nan, prey px, water bearing or nan, water px); bearing in the head frame."""
    out = np.full((ret.shape[0], E.NE, 4), np.nan)
    xs = np.arange(W) + 0.5
    for e in range(E.NE):
        yaw0, f = E.EYES[e][2], TABLE[e, 7]
        for kind, mask in ((0, ret[:, e] < 15), (2, ret[:, e] > 200)):
            n = mask.sum((1, 2))
            cx = (mask.sum(1) * xs).sum(1) / np.maximum(n, 1)
            out[:, e, kind] = np.where(n > 0, yaw0 + np.degrees(np.arctan((W / 2 - cx) / f)), np.nan)
            out[:, e, kind + 1] = n
    return out


class Needs:
    def __init__(self, hunger, thirst, refill=1.0):
        self.hunger, self.thirst, self.refill = np.asarray(hunger, float).copy(), np.asarray(thirst, float).copy(), refill

    def tick(self, head, prey):
        self.hunger = np.clip(self.hunger + HUNGER_S * DT, 0, 1)
        self.thirst = np.clip(self.thirst + THIRST_S * DT, 0, 1)
        at_water = np.hypot(head[0] - WATER_XY[0], head[1] - WATER_XY[1]) < WATER_R
        self.thirst = np.where(at_water, self.thirst * (1 - self.refill), self.thirst)  # refill = 1: drinking empties the need
        caught = np.hypot(*(head[:2] - prey[:2])) < CATCH_MM
        self.hunger = np.where(caught, 0.0, self.hunger)
        return at_water


class Policy:
    """the fixed drive rules; sign = +1 normally, -1 for a rule's mutant (that rule turns the other way); off = rules disabled."""
    def __init__(self, needs, flip=(), off=()):
        self.needs, self.flip, self.off = needs, set(flip), set(off)
        self.area = [np.zeros(B)] * 5
        self.fear_left, self.fear_dir = np.zeros(B, int), np.zeros(B)

    def s(self, rule):
        return -1.0 if rule in self.flip else 1.0

    def __call__(self, ret, gaze):
        b = blobs(ret)
        area = np.nansum(b[:, :, 1], 1)
        grow = area > np.maximum(self.area[0], 1) * FEAR_GROW
        self.area = self.area[1:] + [area]
        prey_am = np.isfinite(b[:, AM, 0]).any(1)
        side = np.nanmean(np.where(np.isfinite(b[:, 2:, 0]), b[:, 2:, 0], np.nan), 1) if np.isfinite(b[:, 2:, 0]).any() else np.full(B, np.nan)
        prey_bear = np.where(prey_am, np.nanmean(b[:, AM, 0], 1) if np.isfinite(b[:, AM, 0]).any() else 0, side)
        water = np.nanmean(np.where(np.isfinite(b[:, :, 2]), b[:, :, 2], np.nan), 1) if np.isfinite(b[:, :, 2]).any() else np.full(B, np.nan)
        speed, turn, rule = np.zeros(B), np.zeros(B), np.array(['idle'] * B, object)
        h, t = self.needs.hunger, self.needs.thirst
        new_fear = grow & (area > 20) & (self.fear_left == 0) & ('fear' not in self.off)
        self.fear_dir = np.where(new_fear, -np.sign(np.nan_to_num(prey_bear, nan=1.0)) * self.s('fear'), self.fear_dir)
        self.fear_left = np.where(new_fear, FEAR_STEPS, np.maximum(self.fear_left - 1, 0))
        for i in range(B):
            if self.fear_left[i] > 0:
                speed[i], turn[i], rule[i] = 1.0, self.fear_dir[i], 'fear'
            elif h[i] >= 0.2 and h[i] >= t[i] and np.isfinite(prey_bear[i]) and 'hunt' not in self.off:
                if prey_am[i]:
                    speed[i], turn[i], rule[i] = 0.6 * h[i], np.clip(gaze[i] / 2.0, -1, 1), 'orient'
                elif 'instinct' not in self.off:
                    speed[i], turn[i], rule[i] = 0.3, np.sign(prey_bear[i]) * self.s('instinct'), 'instinct'
            elif t[i] >= 0.5 and 'drink' not in self.off:
                if np.isfinite(water[i]):
                    speed[i], turn[i], rule[i] = 0.8, np.clip(water[i] / 20.0, -1, 1) * self.s('drink'), 'drink'
                else:
                    speed[i], turn[i], rule[i] = 0.3, 0.6, 'scan'
        return speed, turn, rule


def episode(net, rec, T, env, needs, policy_kw=None, view=None):
    """one batch: brain on the principal eyes, Policy on the body. Returns per-step truth: head xy, yaw, prey xy, rules, needs."""
    reset, step = net
    reset()
    pol = Policy(needs, **(policy_kw or {}))
    eyes = L.Eyes(np.zeros(B), np.zeros(B), T, view=view, env=env)
    gaze, eye_v, prev = np.zeros(B), np.zeros(B), None
    ctrl = eyes.ctrl
    rec_ = {'head': [], 'yaw': [], 'prey': [], 'rule': [], 'thirst': [], 'gaze': []}
    for k in range(T):
        now = L.receptors(rec, eyes.sense(gaze, ctrl))
        q = eyes.q
        needs.tick(q[:3], q[-7:-4])
        speed, turn, rule = pol(eyes.ret, gaze)
        ctrl = gait(k * DT, speed, turn).T
        rec_['head'].append(q[:2].copy()); rec_['yaw'].append(np.array([yaw(q[:, i]) for i in range(B)]))
        rec_['prey'].append(q[-7:-5].copy()); rec_['rule'].append(rule); rec_['thirst'].append(needs.thirst.copy()); rec_['gaze'].append(gaze.copy())
        d = step(L.P.motion(prev, now).astype(np.float32)) if prev is not None else np.zeros(B)
        prev = now
        eye_v = eye_v + L.P.KA * d
        gaze = gaze + eye_v * DT
    eyes.close()
    return {k: np.array(v) for k, v in rec_.items()}


def prey_at(az, dist, v_dir=None, speed=0.0):
    """env callable: still (or moving at speed mm/s along v_dir deg, head frame) prey at az deg, dist mm from AM_L."""
    def env(head, quat):
        rows = []
        for a in np.broadcast_to(az, (B,)):
            p = E.place(head, quat, 'AM_L', a, 0, dist)
            vd = np.radians(a if v_dir is None else v_dir)
            rows.append((*p, speed * np.cos(vd), speed * np.sin(vd), 0.0))
        return rows
    return env


def bearing(r):
    """prey bearing from the head relative to its yaw, deg, per step and animal."""
    b = np.degrees(np.arctan2(r['prey'][..., 1, :] - r['head'][..., 1, :], r['prey'][..., 0, :] - r['head'][..., 0, :]))
    return wrap(b - r['yaw'])


def check_needs():
    n = Needs(np.zeros(2), np.full(2, 0.6))
    for _ in range(100):
        n.tick(np.array([[WATER_XY[0], 0.0], [WATER_XY[1], 0.0]]), np.array([[99.0, 99.0], [99.0, 99.0]]))
    drink = n.thirst[0] < 0.01 and n.thirst[1] > 0.6 and n.hunger[0] > 0
    m = Needs(np.zeros(1), np.full(1, 0.6), refill=-1.0)
    m.tick(np.array([[WATER_XY[0]], [WATER_XY[1]]]), np.array([[99.0], [99.0]]))
    mut = m.thirst[0] < 0.01
    ok = drink and not mut
    print(f"{'PASS' if ok else 'FAIL'} N needs: thirst at the water {n.thirst[0]:.3f}, away {n.thirst[1]:.3f} (rose from 0.6), hunger rose to {n.hunger[0]:.4f}; refill-sign mutant drinks: {mut}")
    return ok


def check():
    bad = int(not check_needs())
    rec = L.P.retina(np.random.default_rng(0))
    _, net = L.G.train('learn', 300, rec, 2, keep=True)
    p = L.pathway(net)
    res = {}
    az = np.where(np.arange(B) % 2, 1, -1) * np.linspace(22, 30, B)
    hungry = lambda: Needs(np.ones(B), np.zeros(B))
    # I instinct
    got = {}
    for arm, kw in (('reflex', {}), ('noreflex', {'off': ('instinct',)}), ('mutant', {'flip': ('instinct',)})):
        r = episode(p, rec, 300, prey_at(az, 20.0), hungry(), kw)  # 20 mm: 4.5 px in a 70 deg secondary eye (191 mm would be < 1 px)
        got[arm] = float(np.mean(np.abs(bearing(r)[-1] - r['gaze'][-1]) < 3))
    res['instinct_fovea_frac'] = got
    ok = got['reflex'] >= 0.8 and got['noreflex'] <= 0.2 and got['mutant'] <= 0.2
    print(f"{'PASS' if ok else 'FAIL'} I instinct: prey in the principal eyes after 3 s: reflex {got['reflex']:.0%}, no reflex {got['noreflex']:.0%}, mutant {got['mutant']:.0%}")
    bad += not ok
    # H hunger
    sp = L.P.speeds(np.random.default_rng(31), 6, 9, B)
    env = lambda head, quat: L.env_rows(head, quat, np.random.default_rng(37).uniform(-0.3, 0.3, B), sp)
    got = {}
    for arm, h in (('hungry', 1.0), ('sated', 0.1)):
        r = episode(p, rec, 500, env, Needs(np.full(B, h), np.zeros(B)))
        bchg = wrap(bearing(r)[-1] + r['yaw'][-1] - bearing(r)[0] - r['yaw'][0])
        got[arm] = {'yaw_frac': float(np.median(np.abs(wrap(r['yaw'][-1] - r['yaw'][0])) / np.maximum(np.abs(bchg), 1e-6))),
                    'moved_mm': float(np.median(np.hypot(*(r['head'][-1] - r['head'][0]))))}
    res['hunger'] = got
    ok = got['hungry']['yaw_frac'] >= 0.5 and got['sated']['yaw_frac'] < 0.1 and got['sated']['moved_mm'] < 1
    print(f"{'PASS' if ok else 'FAIL'} H hunger: body yaw / prey bearing change hungry {got['hungry']['yaw_frac']:.0%}, sated {got['sated']['yaw_frac']:.0%} "
          f"(moved {got['sated']['moved_mm']:.2f} mm)")
    bad += not ok
    # F fear
    got = {}
    for arm, v, kw in (('looming', 100.0, {}), ('receding', -100.0, {}), ('mutant', 100.0, {'flip': ('fear',)})):
        def env(head, quat, v=v):
            rows = prey_at(np.full(B, 20.0), 60.0)(head, quat)
            return [(x, y, z, -v * (x - head[0]) / np.hypot(x - head[0], y - head[1]), -v * (y - head[1]) / np.hypot(x - head[0], y - head[1]), 0.0) for x, y, z, *_ in rows]
        r = episode(p, rec, 40, env, Needs(np.zeros(B), np.zeros(B)), kw)  # 0.4 s: the object stops 20 mm short, it never passes the spider
        got[arm] = float(np.median(np.abs(bearing(r)[-1]) - np.abs(bearing(r)[0])))
    res['fear_turned_away_deg'] = got
    ok = got['looming'] >= 15 and abs(got['receding']) < 5 and got['mutant'] < 0
    print(f"{'PASS' if ok else 'FAIL'} F fear: turned away from the object in 0.4 s (|bearing| change): looming {got['looming']:+.1f} deg (>= 15), "
          f"receding {got['receding']:+.1f} (< 5), mutant {got['mutant']:+.1f} (< 0)")
    bad += not ok
    # W water
    got = {}
    far = prey_at(np.full(B, 180.0), 150.0)  # the prey out of sight behind
    for arm, t, kw in (('thirsty', 0.9, {}), ('quenched', 0.0, {}), ('mutant', 0.9, {'flip': ('drink',)})):
        r = episode(p, rec, 800, far, Needs(np.zeros(B), np.full(B, t)), kw)
        d = np.hypot(r['head'][:, 0] - WATER_XY[0], r['head'][:, 1] - WATER_XY[1])
        got[arm] = {'reached': float(np.mean(d.min(0) < WATER_R)), 'thirst_end': float(np.median(r['thirst'][-1]))}
    res['water'] = got
    ok = got['thirsty']['reached'] >= 0.7 and got['thirsty']['thirst_end'] < 0.2 and got['quenched']['reached'] <= 0.1 and got['mutant']['reached'] <= 0.1
    print(f"{'PASS' if ok else 'FAIL'} W water: reached the drop thirsty {got['thirsty']['reached']:.0%} (thirst after {got['thirsty']['thirst_end']:.2f}), "
          f"not thirsty {got['quenched']['reached']:.0%}, mutant {got['mutant']['reached']:.0%}")
    bad += not ok
    net.close()
    (L.G.ROOT / '.work/agent/spider/drives.json').write_text(json.dumps(res, indent=2) + '\n')
    if bad:
        sys.exit(f'FAIL {bad} drive checks')
    print('PASS spider drives: hunger, thirst, fear, instinct, water seeking')


if __name__ == '__main__':
    {'check': check, 'needs': check_needs}[sys.argv[1]]()
