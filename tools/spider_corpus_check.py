#!/usr/bin/env python3
"""Validate the isolated spider literature packet produced by spider_corpus.py."""
import argparse
import json
from pathlib import Path

from spider_corpus import DEFAULT_OUT, check

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
args = parser.parse_args()
print(json.dumps(check(args.out), sort_keys=True))
