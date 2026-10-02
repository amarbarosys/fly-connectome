"""Test dopamine-gated KC->MBON plasticity on the full FlyWire graph, on CPU."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from agent.mb_plasticity import MBPlasticity, odor_drive, sparsen
from agent.neural_connectome import ROOT, CPUBrain, digest

OUT = ROOT/'.work/agent/mb-plasticity'
ODOR = np.array([[1, 1, 0, 0, 0, 0, 0, 0]], np.float32)
EXCHANGES = 30


def expose(brain, cells, plastic=None):
    """Odour for EXCHANGES body exchanges from rest; summed MBON trace and KCs that ever fired."""
    brain.reset()
    mbon = 0.0
    for _ in range(EXCHANGES):
        brain.step(ODOR)
        if plastic is not None:
            plastic.observe()
        mbon += float(brain.trace[cells['MBON'], 0].sum())
    return mbon, int(brain.ever[cells['KC'], 0].sum())


def verify():
    checks = {}
    brain = CPUBrain(flies=1)
    original = brain.graph.data.copy()
    plastic = MBPlasticity(brain)
    cells, s = plastic.cells, plastic.summary()
    checks['populations'] = {'pass': s['cells']['KC'] == 5177 and s['cells']['MBON'] == 96
        and s['cells']['PAM'] > 0 and s['cells']['PPL1'] > 0 and s['kc_mbon_edges'] > 0,
        'cells': s['cells'], 'kc_mbon_edges': s['kc_mbon_edges'],
        'excluded_nonpositive': s['kc_mbon_edges_excluded_nonpositive']}

    frozen_mbon, kc_active = expose(brain, cells, plastic)
    checks['no_change_without_dopamine'] = {'pass': np.array_equal(brain.graph.data, original)}
    checks['odor_reaches_kcs'] = {'pass': kc_active > 0 and plastic.e.max() > 0,
        'kcs_fired': kc_active, 'eligible_edges': int((plastic.e > 0).sum())}

    drive = odor_drive(['DA1'])
    orn_nodes = np.flatnonzero(drive[:, 0])
    # Verify drive node identity against the annotation roster (cell-type labels
    # include both sides under the ORN_DA1 token).
    roots = np.load(ROOT/'data/proofread_root_ids_783.npy')
    active_roots = {int(roots[i]) for i in orn_nodes}
    import csv
    annotation_path = ROOT/'data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv'
    active_rows = [row for row in csv.DictReader(annotation_path.open(), delimiter='\t')
                   if int(row['root_id']) in active_roots]
    checks['odor_drive_da1_targets_olfactory_orns'] = {
        'pass': bool(orn_nodes.size) and len(active_rows) == len(orn_nodes) and all(
            row['cell_class'] == 'olfactory' and row['cell_type'] == 'ORN_DA1'
            for row in active_rows), 'active_nodes': len(orn_nodes)}

    sparse_brain = CPUBrain(flies=1)
    sparse_cells = MBPlasticity(sparse_brain).cells
    k = sparsen(sparse_brain, sparse_cells, fraction=1.0)
    sparse_max = 0
    sparse_brain.reset()
    for _ in range(EXCHANGES):
        sparse_brain.step(ODOR)
        sparse_max = max(sparse_max, int(sparse_brain.spikes[sparse_cells['KC'], 0].sum()))
    checks['sparsen_caps_kc_spikes'] = {'pass': sparse_max <= k and k == len(sparse_cells['KC']) - 1,
        'maximum_kc_spikes_per_tick': sparse_max, 'k': k, 'kc_count': len(sparse_cells['KC'])}
    class NonCPU:
        pass
    rejected = False
    try:
        sparsen(NonCPU(), sparse_cells)
    except TypeError:
        rejected = True
    checks['sparsen_rejects_non_cpu_brain'] = {'pass': rejected}

    # Compare the default no-extra-drive/no-sparsity CPUBrain path against the
    # pre-change implementation loaded from the committed source snapshot.
    import importlib.util
    import subprocess
    old_source = subprocess.check_output(
        ['git', 'show', 'HEAD:agent/neural_connectome.py'], cwd=ROOT, text=True)
    import tempfile
    fd, old_name = tempfile.mkstemp(prefix='_neural_connectome_prechange_', suffix='.py', dir=ROOT/'agent')
    old_path = Path(old_name)
    try:
        old_path.write_text(old_source)
        spec = importlib.util.spec_from_file_location('neural_connectome_prechange', old_path)
        old_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old_module
        spec.loader.exec_module(old_module)
        before, after = old_module.CPUBrain(flies=1), CPUBrain(flies=1)
        fixed_input = np.array([[1., 0., 0., 0., 0., 0., 0., 0.]], np.float32)
        byte_identical = True
        for _ in range(5):
            old_features = np.asarray(before.step(fixed_input)['features'], dtype=np.float32)
            new_features = np.asarray(after.step(fixed_input, extra_drive=None)['features'], dtype=np.float32)
            byte_identical &= old_features.tobytes() == new_features.tobytes()
        checks['default_cpu_step_matches_prechange_five_steps'] = {'pass': byte_identical}
    finally:
        old_path.unlink(missing_ok=True)

    plastic.dopamine(+1, 'reward after odour')
    changed = np.flatnonzero(brain.graph.data != original)
    ratio = brain.graph.data[plastic.pos]/plastic.w0
    checks['reward_potentiates_block_only'] = {'pass': len(changed) > 0 and np.isin(changed, plastic.pos).all()
        and ratio.min() >= 1 and ratio.max() <= plastic.p.w_max_ratio+1e-6,
        'edges_changed': int(len(changed)), 'max_ratio': float(ratio.max())}
    paired_mbon, _ = expose(brain, cells)
    checks['reward_changes_mbon_response'] = {'pass': paired_mbon != frozen_mbon,
        'mbon_trace_frozen': frozen_mbon, 'mbon_trace_after_reward': paired_mbon}

    plastic.detach()
    checks['detach_restores_exact_bytes'] = {'pass': brain.graph.data.tobytes() == original.tobytes()}
    plastic.attach()
    checks['attach_restores_learned'] = {'pass': np.array_equal(brain.graph.data[plastic.pos], plastic.w_learned)}
    logged = sum(event['abs_delta'] for event in plastic.log)
    actual = float(np.abs(brain.graph.data[plastic.pos]-plastic.w0).sum())
    checks['log_accounts_for_change'] = {'pass': bool(np.isclose(logged, actual, rtol=1e-5)),
        'logged_abs_delta': logged, 'actual_abs_delta': actual}
    plastic.write_log(OUT/'reward-log.json')

    brain = CPUBrain(flies=1)
    punished = MBPlasticity(brain)
    expose(brain, cells, punished)
    punished.dopamine(-1, 'punishment after odour')
    ratio = brain.graph.data[punished.pos]/punished.w0
    checks['punishment_depresses_within_floor'] = {'pass': bool(ratio.max() <= 1 and (ratio < 1).any()
        and ratio.min() >= punished.p.w_min_ratio-1e-6), 'min_ratio': float(ratio.min())}
    punished.write_log(OUT/'punishment-log.json')

    failed = [name for name, row in checks.items() if not row['pass']]
    report = {'status': 'FAIL' if failed else 'PASS', 'failed': failed, 'checks': checks,
        'exchanges_per_exposure': EXCHANGES,
        'scope': 'mechanics of the ported rule on the real graph: eligibility, bounded potentiation and '
                 'depression of KC->MBON edges only, exact restore, logged changes, and a changed MBON '
                 'response to the paired odour; no behavioural learning claim',
        'source': 'rule ported from ASHR12/FlySweeper flysweeper/plasticity.py (MIT)',
        'sources_sha256': {str(p.relative_to(ROOT)): digest(p)
                           for p in (ROOT/'agent/mb_plasticity.py', Path(__file__))},
        'recorded_unix': time.time()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'verification.json').write_text(json.dumps(report, indent=2, default=bool)+'\n')
    for name, row in checks.items():
        print(('PASS ' if row['pass'] else 'FAIL ')+name)
    print('status', report['status'], f'{len(checks)-len(failed)}/{len(checks)}', OUT/'verification.json')
    return report

if __name__ == '__main__':sys.exit(verify()['status'] != 'PASS')
