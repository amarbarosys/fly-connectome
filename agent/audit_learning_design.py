"""Read-only acceptance ceiling and linear-adapter capacity falsifiers."""
import json
from pathlib import Path
import time
import numpy as np
from agent.neural_connectome import ROOT,digest


def audit():
    source=ROOT/'.work/agent/odor-memory/sequential-v2'
    protocol=json.loads((source/'protocol.json').read_text())
    metrics=json.loads((source/'metrics.json').read_text())
    required=protocol['criteria']['minimum_success_improvement']
    controls={name:{'baseline_success':metrics['avoidance'][name]['success_fraction'],
        'maximum_possible_absolute_gain':1-metrics['avoidance'][name]['success_fraction'],
        'required_absolute_gain':required,
        'feasible_on_recorded_fixture':1-metrics['avoidance'][name]['success_fraction']>=required}
        for name in protocol['controls'][1:]}
    rng=np.random.default_rng(260603979);inputs=rng.normal(size=(16,2607))
    weights=rng.normal(size=(2607,5));a=rng.normal(size=(2607,2));b=rng.normal(size=(2,5))
    expanded=inputs@weights+(inputs@a)@b
    merged=inputs@(weights+a@b)
    error=float(np.max(np.abs(expanded-merged)));assert error<1e-10
    assert not controls['untrained_straight']['feasible_on_recorded_fixture']
    report={'status':'PASS','gate_feasibility':'FAIL','recorded_avoidance_gate_unchanged':metrics['status'],
        'controls':controls,'linear_adapter_equivalence_max_abs_error':error,
        'linear_adapter_capacity':'ungated same-feature additive low-rank adapters merge into the existing 2607x5 affine matrix; no new function class',
        'scope':'acceptance arithmetic on saved metrics and synthetic linear algebra, not training or biological evidence',
        'real_targets_refit_or_rescored':False,'active_model_changed':False,
        'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            [source/'metrics.json',source/'protocol.json',ROOT/'agent/neural_connectome.py',Path(__file__)]},
        'recorded_unix':time.time()}
    out=ROOT/'.work/agent/control-audit/learning-design-verification.json'
    out.write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':print(json.dumps(audit()))
