#!/usr/bin/env python3
"""Validate the isolated spider literature packet produced by spider_corpus.py."""
import argparse
import json
from pathlib import Path

from spider_corpus import DEFAULT_OUT, ROOT, check

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
print(json.dumps(check(args.out, ROOT / 'tools/spider_numeric_audit.jsonl'), sort_keys=True))
