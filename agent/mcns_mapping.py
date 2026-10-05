"""Name-based descending-neuron and leg motor-neuron selectors for MCNS."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent import mcns
from agent.neural_connectome import ANNOTATIONS, ROOT

FLYWIRE_ANNOTATIONS = ANNOTATIONS
if not FLYWIRE_ANNOTATIONS.is_file():
    # Worktrees share the checkout's untracked FlyWire annotation asset, but do not inherit it.
    shared_asset = (ROOT.parents[2] / 'data/annotations/flyconnectome-flywire_annotations-a83b277'
                    / 'supplemental_files/Supplemental_file1_neuron_annotations.tsv')
    if shared_asset.is_file():
        FLYWIRE_ANNOTATIONS = shared_asset


def annotation_table():
    """Rows in the same Traced/Anchor order as mcns.graph() neuron indices."""
    a = pd.read_feather(mcns.DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                        columns=['bodyId', 'status', 'superclass', 'type', 'flywireType',
                                 'somaNeuromere', 'somaSide', 'subclass'])
    return a[a.status.isin(['Traced', 'Anchor'])].reset_index(drop=True)


def descending_name_sets():
    """MCNS graph indices by FlyWire-compatible DN type and side, plus coverage counts."""
    m = annotation_table()
    fw = pd.read_csv(FLYWIRE_ANNOTATIONS, sep='\t', usecols=['super_class', 'cell_type', 'side'])
    fw = fw[fw.super_class == 'descending'].copy()
    fw['cell_type'] = fw.cell_type.fillna('')
    m['type'] = m.type.fillna('')
    fw_types = set(fw.cell_type)
    mcns_types = set(m.loc[m.superclass == 'descending_neuron', 'type'])
    turning_types = {'DNa01', 'DNa02'}
    forward_types = {t for t in fw_types if t.startswith(('DNg', 'oDN1'))}
    sets = {'turn_left': [], 'turn_right': [], 'forward': [],
            'forward_left': [], 'forward_right': [], 'forward_center': []}
    selected = m.superclass == 'descending_neuron'
    for i, row in m.loc[selected].iterrows():
        if row.type not in fw_types:
            continue
        if row.type in turning_types:
            key = {'L': 'turn_left', 'R': 'turn_right'}.get(row.somaSide)
            if key:
                sets[key].append(int(i))
        elif row.type in forward_types:
            sets['forward'].append(int(i))
            key = {'L': 'forward_left', 'R': 'forward_right', 'M': 'forward_center'}.get(row.somaSide)
            if key:
                sets[key].append(int(i))
    counts = {}
    for name, wanted in (('turning', turning_types), ('forward', forward_types)):
        frows = fw[fw.cell_type.isin(wanted)]
        mrows = m[(m.superclass == 'descending_neuron') & m.type.isin(wanted)]
        matched = wanted & mcns_types
        counts[name] = {'flywire_types': len(wanted), 'matched_types': len(matched),
                        'unmatched_types': sorted(wanted - matched),
                        'flywire_cells': len(frows), 'matched_mcns_cells': len(mrows),
                        'flywire_cells_of_matched_types': int(frows.cell_type.isin(matched).sum())}
    counts['all_descending'] = {'flywire_types': len(fw_types - {''}),
                                'matched_types': len((fw_types - {''}) & mcns_types),
                                'unmatched_types': sorted((fw_types - {''}) - mcns_types),
                                'flywire_cells': len(fw),
                                'matched_mcns_cells': int((m.superclass == 'descending_neuron').sum())}
    return {name: np.asarray(ids, np.int32) for name, ids in sets.items()}, counts


def leg_motor_groups():
    """Annotated leg MN graph indices by actuator-side leg (front/middle/hind) and soma side."""
    m = annotation_table()
    selected = ((m.superclass == 'vnc_motor') & m.somaNeuromere.isin(['T1', 'T2', 'T3'])
                & m.subclass.isin(['fl', 'ml', 'hl']))
    leg_name = {'fl': 'front', 'ml': 'middle', 'hl': 'hind'}
    groups = {f'{leg}_{side}': [] for leg in leg_name.values() for side in ('L', 'R')}
    for i, row in m.loc[selected].iterrows():
        key = groups.get(f"{leg_name[row.subclass]}_{row.somaSide}")
        if key is not None:
            key.append(int(i))
    counts = {'annotated_leg_motor_neurons': int(selected.sum()),
              'by_leg_neuromere': {leg_name[k]: int(v) for k, v in m.loc[selected, 'subclass'].value_counts().items()},
              'by_side': {str(k): int(v) for k, v in m.loc[selected, 'somaSide'].value_counts().items()},
              'groups': {k: len(v) for k, v in groups.items()},
              'annotated_types': int(m.loc[selected, 'type'].nunique())}
    return {name: np.asarray(ids, np.int32) for name, ids in groups.items()}, counts


def motor_mapping_coverage():
    """Compare strict actuator-axis labels with the broader named-muscle antagonist map."""
    m = annotation_table()
    selected = ((m.superclass == 'vnc_motor') & m.somaNeuromere.isin(['T1', 'T2', 'T3'])
                & m.subclass.isin(['fl', 'ml', 'hl']))
    types = set(m.loc[selected, 'type'].dropna())
    strict_axes = ('coxa yaw', 'coxa pitch', 'coxa roll', 'trochanterofemur pitch',
                   'trochanterofemur roll', 'tibia pitch', 'tarsus1 pitch')
    # These names identify opposing muscle actions, but not the remaining coxa and roll axes.
    muscle_groups = {
        'trochanterofemur_pitch': sorted(t for t in types if t in {
            'Tr flexor MN', 'Tr extensor MN', 'Acc. tr flexor MN'}),
        'tibia_pitch': sorted(t for t in types if t in {
            'Ti flexor MN', 'Ti extensor MN', 'Acc. ti flexor MN'}),
        'tarsus1_pitch': sorted(t for t in types if t in {'Ta levator MN', 'Ta depressor MN'}),
    }
    annotated_axes = [axis for axis in strict_axes if any(axis in t.lower() for t in types)]
    return {'approach_1_explicit_axis_names': {'covered_axes': annotated_axes,
                                               'coverage': f'{len(annotated_axes)}/7'},
            'approach_2_antagonist_muscle_names': {'groups': muscle_groups,
                                                    'covered_axes': sorted(muscle_groups),
                                                    'coverage': f'{len(muscle_groups)}/7',
                                                    'unresolved_axes': ['coxa yaw', 'coxa pitch', 'coxa roll',
                                                                        'trochanterofemur roll']},
            'six_leg_channel_coverage': f'{6 * len(muscle_groups)}/42'}


def visual_drive_groups():
    """LC4/LPLC2 looming cells by anatomical side for a FlyWire-compatible sensory drive."""
    m = annotation_table()
    selected = ((m.superclass == 'visual_projection') & m.type.isin(['LC4', 'LPLC2']))
    groups = {side: np.flatnonzero((selected & (m.somaSide == code)).to_numpy()).astype(np.int32)
              for side, code in (('left', 'L'), ('right', 'R'))}
    counts = {'LC4': int((selected & (m.type == 'LC4')).sum()),
              'LPLC2': int((selected & (m.type == 'LPLC2')).sum()),
              'left': len(groups['left']), 'right': len(groups['right'])}
    return groups, counts


def report():
    _, dn = descending_name_sets()
    _, mn = leg_motor_groups()
    _, sensory = visual_drive_groups()
    result = {'descending_match': dn, 'visual_drive': sensory, 'leg_motor_coverage': mn,
              'motor_mapping': motor_mapping_coverage()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    report()
