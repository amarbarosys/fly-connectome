"""Brain v0 for the spider body (models/spider.py): an alternating-tetrapod gait generator with two drives.

Not a spider circuit: a stand-in so the brain<->body link (private note SPIDER_BRAIN unit 1) has a controller whose
effect on the body is measurable. Per leg a phase advances at FREQ Hz; the coxa yaw sweeps +-YAW rad about the stance
(forward in the second half of the cycle) and coxa_pitch lifts the foot during the forward sweep. Tetrapod groups
l1 r2 l3 r4 and r1 l2 r3 l4 sit half a cycle apart. Drives: speed in [0, 1] scales the sweep (0: every servo at the
crouch, the spider stands), turn in [-1, 1] shortens the sweep on the side opposite the turn (+ = left, measured: +0.8 turns +112 deg/s).

ctrl layout = the servo order of models/spider.py: pedicel_pitch, pedicel_yaw, then legs l1..l4, r1..r4 with 7 servos
each (coxa_yaw, coxa_pitch, femur, patella, tibia, metatarsus, tarsus). Free-body check: scratchpad sweep of 8 settings in
MuJoCo, forward +x at 11 mm / 0.6 s with LIFT = +0.3, zero speed stands (spider_link check).
"""
import numpy as np

FREQ, YAW, LIFT = 5.0, 0.4, 0.3
NU = 2 + 8 * 7
TETRAPOD = np.array([0, 1, 0, 1, 1, 0, 1, 0], np.float64)
SIDE = np.array([1, 1, 1, 1, -1, -1, -1, -1], np.float64)


def gait(t, speed, turn):
    """Servo targets (F, NU) at time t (s, scalar or (F,)) for drives speed, turn ((F,) each)."""
    t, speed, turn = (np.asarray(x, np.float64) for x in (t, speed, turn))
    ph = (FREQ * t[..., None] + 0.5 * TETRAPOD) % 1.0  # (F, 8)
    sweep = (1.0 + turn[..., None] * SIDE) * speed[..., None]
    u = np.zeros(speed.shape + (NU,))
    leg = u[..., 2:].reshape(speed.shape + (8, 7))
    leg[..., 0] = -SIDE * YAW * sweep * -np.cos(2 * np.pi * ph)
    leg[..., 1] = LIFT * speed[..., None] * np.maximum(0.0, -np.sin(2 * np.pi * ph))
    return u.astype(np.float32)
