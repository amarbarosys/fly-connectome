"""Build a receipt-backed capability ledger without rerunning old experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.neural_connectome import ROOT, digest

# Claim boundaries are explicit; a receipt's PASS is never blanket validation.
CLAIMS = (
    ('neural_interface', 'assets/neural-gate.json', 'Recorded GPU/CPU interface parity and state isolation', 'Sustained behavior and biological function'),
    ('body_interface', 'embodiment-gate.json', 'Short neural-to-command/body causal interface', 'Stable locomotion and task competence'),
    ('decoder_imitation', 'assets/training.json', 'Held-out teacher imitation RMSE vs listed controls', 'Topology advantage, autonomous lifelong learning'),
    ('coarse_vision', 'vision-gate.json', 'Rendered eyes drive coarse neural inputs', 'Object recognition and retinal fidelity'),
    ('wall_turn', 'wall-turn-gate.json', 'Engineered contact-turn reproduction', 'Learned avoidance and general recovery'),
    ('worker_memory', 'memory-gate.json', 'Bounded worker allocation and parity measurement', 'Long-run leak freedom and population capacity'),
    ('airborne_baseline', 'basic-flight/flight-gate.json', 'Short pretrained airborne rollout', 'Connectome flight, takeoff and landing'),
)


def generate(replay: Path, output: Path):
    claims = []
    for claim_id, relative, supports, missing in CLAIMS:
        path = ROOT / '.work/neural-flies' / relative
        item = {'id': claim_id, 'receipt': str(path), 'supports': supports,
                'unverified': missing, 'runtime_rerun_this_task': False}
        if path.exists():
            receipt = json.loads(path.read_text())
            source_checks = {}
            for source, expected in receipt.get('source_hashes', {}).items():
                current = ROOT / source
                source_checks[source] = 'match' if current.exists() and digest(current) == expected else 'mismatch_or_missing'
            item.update(status='RECORDED_RECEIPT_INSPECTED', receipt_sha256=digest(path),
                        receipt_result=receipt, source_checks=source_checks,
                        current_source_binding='partial_checks_only' if source_checks else 'not_established')
        else:
            item['status'] = 'UNVERIFIED_MISSING_RECEIPT'
        claims.append(item)
    metrics = replay / 'metrics.json'
    result = json.loads(metrics.read_text())
    if result.get('status') != 'PASS':
        raise ValueError('replay gate did not pass')
    claims.append({'id': 'cpu_neural_replay', 'status': 'FRESH_CPU_GATE_PASS',
        'receipt': str(metrics.resolve()), 'receipt_sha256': digest(metrics),
        'runtime_rerun_this_task': True, 'supports': result,
        'unverified': 'MuJoCo/body state, GPU worker, readout learning, motivation and full episode replay'})
    passport = {'schema': 'fly-world.evidence-passport.v1', 'claims': claims,
                'rule': 'Recorded results do not certify current code or biological validity',
                'generator_sha256': digest(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(passport, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'claims': len(claims), 'output': str(output),
        'missing_receipts': sum(x['status'] == 'UNVERIFIED_MISSING_RECEIPT' for x in claims)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    generate(args.replay, args.out)
