"""numpy twin of world.mojo (the oracle): height and gradient of the endless seeded terrain, and the check that the
Mojo function matches it. float32 throughout, the same operation order as the Mojo.

usage: python -m agent.body_engine.world diff [N]   (CPU build of world.mojo vs numpy; rel err <= 1e-5 of scale)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

F = np.float32
R = F(0.70710678)
GX = np.array([1, -1, 0, 0, R, -R, R, -R], F)
GY = np.array([0, 0, 1, -1, R, -R, -R, R], F)
DEFAULT = (0.0, 0.0, 0.3, 8.0, 4, 1, 2.0, 0.5)  # slope x y, amplitude mm, wavelength mm, octaves, seed, lacunarity, gain


def grad(ix, iy, seed):
    with np.errstate(over='ignore'):
        h = ix.astype(np.int32).view(np.uint32) * np.uint32(0x8DA6B343) ^ iy.astype(np.int32).view(np.uint32) * np.uint32(0xD8163841) \
            ^ np.uint32(seed) * np.uint32(0xCB1AB31F)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0x85EBCA6B)
        h ^= h >> np.uint32(16)
    k = h >> np.uint32(29)
    return GX[k], GY[k]


def noise(x, y, seed):
    x0, y0 = np.floor(x), np.floor(y)
    ix, iy = x0.astype(np.int32), y0.astype(np.int32)
    fx, fy = x - x0, y - y0
    ga, gb, gc, gd = grad(ix, iy, seed), grad(ix + 1, iy, seed), grad(ix, iy + 1, seed), grad(ix + 1, iy + 1, seed)
    a = ga[0] * fx + ga[1] * fy
    b = gb[0] * (fx - 1) + gb[1] * fy
    c = gc[0] * fx + gc[1] * (fy - 1)
    d = gd[0] * (fx - 1) + gd[1] * (fy - 1)
    u, du = fx * fx * fx * (fx * (fx * 6 - 15) + 10), 30 * fx * fx * (fx - 1) * (fx - 1)
    v, dv = fy * fy * fy * (fy * (fy * 6 - 15) + 10), 30 * fy * fy * (fy - 1) * (fy - 1)
    k = a - b - c + d
    kx, ky = ga[0] - gb[0] - gc[0] + gd[0], ga[1] - gb[1] - gc[1] + gd[1]
    n = a + (b - a) * u + (c - a) * v + k * u * v
    nx = ga[0] + (gb[0] - ga[0]) * u + (gc[0] - ga[0]) * v + kx * u * v + du * ((b - a) + k * v)
    ny = ga[1] + (gb[1] - ga[1]) * u + (gc[1] - ga[1]) * v + ky * u * v + dv * ((c - a) + k * u)
    return n, nx, ny


def height_grad(x, y, p=DEFAULT):
    x, y, p = np.asarray(x, F), np.asarray(y, F), np.asarray(p, F)
    h, gx, gy = p[0] * x + p[1] * y, np.full_like(x, p[0]), np.full_like(x, p[1])
    amp, f = p[2], F(1) / p[3]
    for o in range(int(p[4])):
        n, nx, ny = noise(x * f, y * f, int(p[5]) + o)
        h, gx, gy = h + amp * n, gx + amp * f * nx, gy + amp * f * ny
        amp, f = amp * p[7], f * p[6]
    return h, gx, gy


def diff(n=100_000, p=DEFAULT):
    out = Path('.work/agent/body_engine/world')
    out.mkdir(parents=True, exist_ok=True)
    xy = np.random.default_rng(0).uniform(-2e3, 2e3, (n, 2)).astype(F)
    np.concatenate([xy.ravel(), np.asarray(p, F)]).tofile(out / 'in.bin')
    binary = out / 'world'
    subprocess.run([str(Path.home() / '.local/bin/mojo'), 'build', str(Path(__file__).with_name('world.mojo')), '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(out / 'in.bin'), str(out / 'out.bin'), str(n)], check=True)
    got = np.fromfile(out / 'out.bin', F).reshape(n, 3)
    bad = 0
    for i, (name, want) in enumerate(zip(('height', 'dh/dx', 'dh/dy'), height_grad(xy[:, 0], xy[:, 1], p))):
        err = float(np.abs(got[:, i] - want).max() / max(np.abs(want).max(), 1e-30))
        print(f'{"PASS" if err <= 1e-5 else "FAIL"} {name:6} max rel err {err:.2e} over {n} points')
        bad += err > 1e-5
    e = F(1e-2)  # the gradient against central differences of the height (both twins share one derivation)
    fd = [(height_grad(xy[:, 0] + e * (k == 0), xy[:, 1] + e * (k == 1), p)[0] - height_grad(xy[:, 0] - e * (k == 0), xy[:, 1] - e * (k == 1), p)[0]) / (2 * e) for k in (0, 1)]
    _, gx, gy = height_grad(xy[:, 0], xy[:, 1], p)
    err = float(max(np.abs(fd[0] - gx).max(), np.abs(fd[1] - gy).max()) / np.abs(np.r_[gx, gy]).max())
    print(f'{"PASS" if err <= 1e-2 else "FAIL"} gradient vs central difference (h 1e-2 mm, float32): max rel err {err:.2e}')
    bad += err > 1e-2
    if bad:
        sys.exit(f'FAIL world diff: {bad} checks')
    print('PASS world diff')


if __name__ == '__main__':
    diff(*map(int, sys.argv[2:3])) if sys.argv[1] == 'diff' else sys.exit('usage: world diff [N]')
