"""Pixels -> photoreceptors for the male CNS brain: each R1-6 photoreceptor samples an equirectangular luminance image at
its own viewing direction (agent/eye_map.py, flyverse column method, azimuth + = left), and darkening against the
background becomes its Poisson rate. This replaces the legacy pooled-eye interface (one brightness per eye, injected into
L1-L3 uniformly: static position lost). The lamina responds from 5-10 Hz (tools/shiu_dim_test.py on FlyWire).
"""
import numpy as np

from agent.eye_map import PATH, annotations

RMAX, BINS, BG = 150.0, 5, 0.6  # Hz at full darkening, rate levels per image, background luminance


LAMINA = r'^(R[1-6]|L[1-3]$)'  # photoreceptors plus their lamina targets L1-L3 (each column's own luminance)


def receptors(types=r'^R[1-6]'):
    """MCNS cells of `types` (regex on type) with a viewing direction, and their (azimuth, elevation) in degrees."""
    m = np.load(PATH)
    pr = annotations()['type'].fillna('').str.match(types).to_numpy() & ~np.isnan(m['az'])
    ids = np.flatnonzero(pr)
    return ids, m['az'][ids], m['el'][ids]


LEFT_GAIN = 1.6  # tools/fly_eye_calibrate.py: smallest left gain whose LC4/LPLC2 output on a calibration disc (az +-55, el +35) matches the right's


def eye_gain(ids):
    """per-cell rate gain: LEFT_GAIN on the left eye, 1 on the right. The male CNS left path runs weaker (fewer placed L3,
    10-20 % fewer synapses, LPLC2 82k vs 111k input synapses) and dies at the LC4/LPLC2 threshold (disc 5 vs 38 spikes at gain 1);
    equal cell counts were not enough (18 vs 38), so the gain is calibrated on a stimulus no test uses."""
    side = annotations()['somaSide'].fillna('').str[:1].str.upper().to_numpy()[ids]
    return np.where(side == 'L', LEFT_GAIN, 1.0)


def eye_region(az, el):
    """left-eye field region per direction (tools/fly_eye_calibrate_region.py): frontal/lateral split at |az| 72.5, lower/upper at el 5."""
    return np.where(np.abs(az) < 72.5, 'frontal', 'lateral').astype(object) + np.where(el < 5, '-lower', '-upper')


def eye_gain_region(ids, az, el):
    """per-cell rate gain from the per-region calibration: the region's gain on the left eye, 1 on the right."""
    import json
    from agent.neural_connectome import ROOT
    g = json.loads((ROOT / '.work/agent/shiu/eye-gain-region.json').read_text())
    side = annotations()['somaSide'].fillna('').str[:1].str.upper().to_numpy()[ids]
    return np.where(side == 'L', np.vectorize(lambda r: g[r]['gain'])(eye_region(az, el)), 1.0)


def sample(img, az, el):
    """luminance at each direction; img rows = elevation +90..-90, columns = azimuth +180..-180 (left to right)."""
    h, w = img.shape
    r = np.clip(((90 - el) / 180 * h).astype(int), 0, h - 1)
    c = np.clip(((180 - az) / 360 * w).astype(int), 0, w - 1)
    return img[r, c]


def drive(img, ids, az, el, gain=None):
    """[(cell ids, Hz)] for ShiuGPU.stimulate: darkening below the background, quantized to BINS levels (0 dropped);
    gain: per-cell rate factor (eye_gain), applied after quantizing."""
    dark = np.clip((BG - sample(img, az, el)) / BG, 0, 1)
    lvl = np.round(dark * BINS).astype(int)
    hz = RMAX * lvl / BINS * (1.0 if gain is None else gain)
    return [(ids[hz == r], float(r)) for r in np.unique(hz[lvl > 0])]


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
