"""Per-class operating point: how much more recurrent input each cell class needs to fire.

Model before sweep. Under CPUBrain dynamics a cell with steady input u per tick settles at
v = u / (1 - 0.85); it fires when v >= 1. With firing rates r (the spike trace, an exponential
average of spikes) the steady input is u = 0.08 * gain * (W @ r). So the input multiplier a cell
needs is m = 0.15 / (0.08 * gain * W @ r). m < 1: already fires. m > 1: silent by that factor.
A class whose median m is far above 1 while its inputs are active cannot be reached by any global
gain that keeps the upstream class sparse; it needs its own gain.
"""
import numpy as np


def required_multiplier(brain, nodes):
    """Per node (rows) and column: input multiplier needed to reach threshold; inf if input <= 0."""
    u = np.float32(.08) * brain.gain * (brain.graph[nodes] @ brain.trace)
    with np.errstate(divide='ignore'):
        return np.where(u > 0, np.float32(.15) / u, np.inf)


def class_report(brain, classes):
    """For each class: median and 25th percentile, over cells with positive input, of the best column."""
    out = {}
    for name, nodes in classes.items():
        best = required_multiplier(brain, nodes).min(axis=1)
        finite = best[np.isfinite(best)]
        out[name] = {'cells': int(len(nodes)), 'with_positive_input': int(len(finite)),
                     'median_multiplier': float(np.median(finite)) if len(finite) else float('inf'),
                     'q25_multiplier': float(np.quantile(finite, .25)) if len(finite) else float('inf')}
    return out
