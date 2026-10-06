"""Claim seeds once: docs/seeds_used.json lists every seed a claim run has used, per experiment. claim() refuses a used
seed (seed luck: the reflex-off 0.54 deg was one) and records the new ones before the run starts.
"""
import json
import sys
from pathlib import Path

USED = Path(__file__).resolve().parents[1] / 'docs/seeds_used.json'


def claim(exp, seeds):
    used = json.loads(USED.read_text())
    clash = sorted(set(seeds) & set(used.get(exp, [])))
    if clash:
        sys.exit(f'FAIL seeds: {exp} already used {clash} ({USED}); pre-register fresh ones')
    used[exp] = sorted(set(used.get(exp, [])) | set(seeds))
    USED.write_text(json.dumps(used, indent=1) + '\n')
