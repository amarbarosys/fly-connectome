"""Pixels -> photoreceptors for the male CNS brain: each R1-6 photoreceptor samples an equirectangular luminance image at
its own viewing direction (agent/eye_map.py, flyverse column method, azimuth + = left), and darkening against the
background becomes its Poisson rate. This replaces the legacy pooled-eye interface (one brightness per eye, injected into
L1-L3 uniformly: static position lost). The lamina responds from 5-10 Hz (tools/shiu_dim_test.py on FlyWire).
"""
import numpy as np

from agent.eye_map import PATH, annotations

RMAX, BINS, BG = 150.0, 5, 0.6  # Hz at full darkening, rate levels per image, background luminance


def receptors():
    """MCNS R1-6 indices with a viewing direction, and their (azimuth, elevation) in degrees."""
    m = np.load(PATH)
    pr = annotations()['type'].fillna('').str.match(r'^R[1-6]').to_numpy() & ~np.isnan(m['az'])
    ids = np.flatnonzero(pr)
    return ids, m['az'][ids], m['el'][ids]


def sample(img, az, el):
    """luminance at each direction; img rows = elevation +90..-90, columns = azimuth +180..-180 (left to right)."""
    h, w = img.shape
    r = np.clip(((90 - el) / 180 * h).astype(int), 0, h - 1)
    c = np.clip(((180 - az) / 360 * w).astype(int), 0, w - 1)
    return img[r, c]


def drive(img, ids, az, el):
    """[(cell ids, Hz)] for ShiuGPU.stimulate: darkening below the background, quantized to BINS levels (0 dropped)."""
    dark = np.clip((BG - sample(img, az, el)) / BG, 0, 1)
    lvl = np.round(dark * BINS).astype(int)
    return [(ids[lvl == k], RMAX * k / BINS) for k in range(1, BINS + 1) if (lvl == k).any()]


def disc(az0, el0=0.0, radius=40.0, h=90, w=180):
    """a dark disc (luminance 0) on the background, equirectangular h x w."""
    el = 90 - (np.arange(h) + .5) * 180 / h
    az = 180 - (np.arange(w) + .5) * 360 / w
    E, A = np.meshgrid(np.radians(el), np.radians(az), indexing='ij')
    u = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)])
    c = np.array([np.cos(np.radians(el0)) * np.cos(np.radians(az0)), np.cos(np.radians(el0)) * np.sin(np.radians(az0)), np.sin(np.radians(el0))])
    img = np.full((h, w), BG)
    img[np.degrees(np.arccos(np.clip(np.tensordot(c, u, 1), -1, 1))) < radius] = 0
    return img
