"""Paired significance for frozen-layout gates: exact McNemar over per-layout outcomes.

Gate receipts report success fractions, which say how far apart two arms landed but not whether
sixteen layouts can carry the difference. Every arm is evaluated on the SAME frozen layouts, so the
comparison is paired and only the discordant layouts carry information. This reads the per-layout
eval files and reports the exact two-sided binomial test on them.

Small n is the point: 15/16 against 11/16 looks decisive and is not.
"""
import json
from math import comb
from pathlib import Path
import time
from agent.neural_connectome import ROOT, digest

BASELINE = ROOT/'.work/agent/matched-baseline'
SEQUENTIAL = ROOT/'.work/agent/odor-memory/calibration-v2'
OUT = ROOT/'.work/agent/paired-significance'


def exact_mcnemar(first, second):
    """Discordant-pair counts and the exact two-sided p for paired binary outcomes."""
    layouts = sorted(set(first) & set(second))
    wins = sum(1 for k in layouts if first[k] and not second[k])
    losses = sum(1 for k in layouts if second[k] and not first[k])
    discordant = wins+losses
    if not discordant:
        return wins, losses, 0, 1.
    tail = sum(comb(discordant, i) for i in range(min(wins, losses)+1))
    return wins, losses, discordant, min(1., 2.*tail/2**discordant)


def outcomes(directory, control, decide):
    results = {}
    for path in sorted(directory.glob(f'eval-*-{control}.json')):
        row = json.loads(path.read_text())
        if row.get('control') != control:
            continue
        results[json.dumps(row['scene'], sort_keys=True)] = bool(decide(row))
    return results


def compare(label, first, second, pair):
    wins, losses, discordant, p = exact_mcnemar(first, second)
    return {'comparison': label, 'arm_successes': sum(first.values()), 'control_successes': sum(second.values()),
            'layouts': len(first), 'arm_only_wins': wins, 'control_only_wins': losses,
            'discordant_pairs': discordant, 'exact_two_sided_p': p,
            'significant_at_0_05': p < .05, 'pair': pair}


def report():
    ingested = lambda row: row['ingested'] >= .05
    avoided = lambda row: row['contact_exchanges'] <= 1
    rows = []
    arms = {a: outcomes(BASELINE/f'gate-headless-{a}', 'trained', ingested)
            for a in ('real', 'shuffled', 'random')}
    for control in ('shuffled', 'random'):
        rows.append(compare(f'baseline: real graph vs {control} graph', arms['real'], arms[control],
                            'matched-baseline trained arms, same frozen layouts'))
    seq = {c: outcomes(SEQUENTIAL, c, avoided)
           for c in ('trained', 'untrained_straight', 'shuffled_reward')}
    for control in ('untrained_straight', 'shuffled_reward'):
        rows.append(compare(f'sequential avoidance: trained vs {control}', seq['trained'], seq[control],
                            'calibration-v2 frozen held-out layouts'))
    for row in rows:
        assert row['layouts'] == 16, f'expected 16 paired layouts, got {row["layouts"]}'
    out = {'status': 'PASS', 'test': 'exact two-sided McNemar on paired per-layout outcomes',
           'why': 'arms share frozen layouts, so only discordant layouts carry information',
           'comparisons': rows,
           'scope': 'significance of per-layout outcome differences only; says nothing about effect '
                    'size in other tasks, other seeds, or a single shuffle realisation',
           'single_shuffle_realisation': True,
           'sources_sha256': {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
           'recorded_unix': time.time()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'significance.json').write_text(json.dumps(out, indent=2)+'\n')
    for row in rows:
        print(f"{row['comparison']:52s} {row['arm_successes']:2d}/16 vs {row['control_successes']:2d}/16  "
              f"discordant {row['arm_only_wins']}/{row['control_only_wins']}  p={row['exact_two_sided_p']:.4f}  "
              f"{'SIGNIFICANT' if row['significant_at_0_05'] else 'not significant'}")
    return out


if __name__ == '__main__':
    report()
