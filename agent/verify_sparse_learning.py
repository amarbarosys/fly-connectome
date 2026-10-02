"""Sparse olfactory code and A-paired reward check at normalized recurrent gains."""
from __future__ import annotations

import csv
import json
import numpy as np

from agent.mb_plasticity import MBPlasticity, mushroom_body_cells
from agent.neural_connectome import ANNOTATIONS, N, ROOT, CPUBrain

ODORS = (('DA1', 'VA1d', 'DL3'), ('VM2', 'DM1', 'VA2'), ('DM2', 'VC1', 'DA2'))
EXCHANGES = 10


def odor_columns():
    roots = np.load(ROOT / 'data/proofread_root_ids_783.npy')
    node_by_root = {int(root): i for i, root in enumerate(roots)}
    wanted = {f'ORN_{g}' for odor in ODORS for g in odor}
    found = {name: [] for name in wanted}
    with open(ANNOTATIONS) as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            name = row['cell_type']
            if row['cell_class'] == 'olfactory' and name in found:
                node = node_by_root.get(int(row['root_id']))
                if node is not None:
                    found[name].append(node)
    drive = np.zeros((N, len(ODORS)), np.float32)
    for column, odor in enumerate(ODORS):
        nodes = np.unique(np.concatenate([found[f'ORN_{g}'] for g in odor])).astype(np.int32)
        if not len(nodes):
            raise ValueError(f'no annotated ORNs for {odor}')
        drive[nodes, column] = 1
    return drive


def jaccard(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def run_gain(gain):
    drive = odor_columns()
    brain = CPUBrain(flies=3, gain_mode='normalized', gain=gain)
    cells = mushroom_body_cells()
    kc, mbon = cells['KC'], cells['MBON']
    plastic = MBPlasticity(brain, fly=0)
    zeros = np.zeros((3, 8), np.float32)

    # Establish all three cumulative KC sets and the pre-reward MBON responses.
    brain.reset()
    for _ in range(EXCHANGES):
        brain.step(zeros, ticks=4, extra_drive=drive)
    active_sets = brain.ever[kc].copy()
    pre_response = brain.trace[mbon].sum(axis=0).copy()
    active_fraction = active_sets.mean(axis=0)
    pair_jaccards = {f'{i}-{j}': jaccard(active_sets[:, i], active_sets[:, j])
                     for i, j in ((0, 1), (0, 2), (1, 2))}
    sparse_pass = bool(np.all((active_fraction > .01) & (active_fraction < .10)
                              ) and all(x < .3 for x in pair_jaccards.values()))

    # Reset neural state, pair A with reward, and allow eligibility to accumulate on A only.
    brain.reset()
    for _ in range(EXCHANGES):
        brain.step(zeros, ticks=4, extra_drive=drive)
        plastic.observe()
    learning_sets = brain.ever[kc].copy()
    active_kc_edges = np.isin(plastic.pre, kc[learning_sets[:, 0]])
    plastic.dopamine(+1, 'reward after A exposure')
    ratio_change = brain.graph.data[plastic.pos] / plastic.w0 - 1
    mean_change = []
    for odor_index in (0, 1):
        edge_mask = np.isin(plastic.pre, kc[learning_sets[:, odor_index]])
        mean_change.append(float(ratio_change[edge_mask].mean()) if edge_mask.any() else 0.0)
    learning_ratio = (mean_change[0] / mean_change[1] if mean_change[1] > 0
                      else (float('inf') if mean_change[0] > 0 else 0.0))
    never_fired_edges = ~np.isin(plastic.pre, kc[learning_sets[:, 0]])
    never_fired_elig = plastic.e[never_fired_edges]
    active_kc_eligibility = plastic.e[active_kc_edges]

    brain.reset()
    for _ in range(EXCHANGES):
        brain.step(zeros, ticks=4, extra_drive=drive)
    post_response = brain.trace[mbon].sum(axis=0)
    response_delta = np.abs(post_response - pre_response)
    return {
        'gain': gain,
        'kc_active_fraction': active_fraction.tolist(),
        'kc_pairwise_jaccard': pair_jaccards,
        'sparse_code_pass': sparse_pass,
        'mean_ratio_change_on_A_active_KC_edges': mean_change[0],
        'mean_ratio_change_on_B_active_KC_edges': mean_change[1],
        'A_over_B_learning_ratio': learning_ratio,
        'learning_pass': bool(learning_ratio > 2),
        'mbon_pre_reward_response': pre_response.tolist(),
        'mbon_post_reward_response': post_response.tolist(),
        'mbon_abs_response_delta': response_delta.tolist(),
        'mbon_A_changes_more_than_B': bool(response_delta[0] > response_delta[1]),
        'mbon_ever_count': brain.ever[mbon].sum(axis=0).tolist(),
        'mbon_trace_sum': brain.trace[mbon].sum(axis=0).tolist(),
        'active_A_KC_to_MBON_edge_count': int(active_kc_edges.sum()),
        'active_A_KC_edge_nonzero_eligibility_count': int(np.count_nonzero(active_kc_eligibility)),
        'active_A_KC_edge_eligibility_max': float(active_kc_eligibility.max()) if active_kc_eligibility.size else 0.0,
        'kc_never_fired_edge_count': int(never_fired_edges.sum()),
        'kc_never_fired_eligibility_max_abs': float(np.abs(never_fired_elig).max()) if never_fired_elig.size else 0.0,
        'kc_never_fired_nonzero_eligibility_edges': int(np.count_nonzero(never_fired_elig)),
        'pass': bool(sparse_pass and learning_ratio > 2 and response_delta[0] > response_delta[1]),
    }


def verify():
    rows = [run_gain(gain) for gain in (30., 35.)]
    result = {'status': 'PASS' if all(row['pass'] for row in rows) else 'FAIL',
              'odors': [list(odor) for odor in ODORS], 'exchanges': EXCHANGES, 'rows': rows}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    raise SystemExit(verify()['status'] != 'PASS')
