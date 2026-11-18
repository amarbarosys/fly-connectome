#!/usr/bin/env bash
# Fetch the FlyWire (FAFB v783) files this repo reads, from the URLs the code and docs name.
# FlyWire data is not redistributed here (data/ is gitignored). Terms and citation: https://codex.flywire.ai
# usage: tools/fetch_flywire.sh
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
ANN=data/annotations/flyconnectome-flywire_annotations-a83b277
ARCH=.work/fetch/flywire_annotations-a83b277.tar.gz
COL=data/column_assignment.csv.gz

# annotations: the release tree named by the directory (flyconnectome/flywire_annotations @ a83b277)
if [ -f "$ANN/supplemental_files/Supplemental_file1_neuron_annotations.tsv" ]; then
  echo "ok $ANN (present)"
else
  mkdir -p "$(dirname "$ARCH")" "$ANN"
  if curl -fL --retry 3 -o "$ARCH" https://github.com/flyconnectome/flywire_annotations/archive/a83b277.tar.gz \
     && tar -xzf "$ARCH" -C "$ANN" --strip-components=1 \
     && [ -f "$ANN/supplemental_files/Supplemental_file1_neuron_annotations.tsv" ]; then
    echo "ok $ANN"
  else
    echo "FAIL fetch annotations: github.com/flyconnectome/flywire_annotations @ a83b277"; fail=$((fail+1))
  fi
fi

# column assignment (agent/spatial_vision.py SOURCE_URL)
if [ -s "$COL" ]; then
  echo "ok $COL (present)"
elif mkdir -p data && curl -fL --retry 3 -o "$COL.part" https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz; then
  mv "$COL.part" "$COL"; echo "ok $COL"
else
  rm -f "$COL.part"; echo "FAIL fetch $COL: storage.googleapis.com flywire-data"; fail=$((fail+1))
fi

# synapse tables and proofread roster: no URL names them in the code, so they are not fetched here
echo "SKIP synapse tables and proofread roster (data/flywire_synapses_783.feather, data/proofread_connections_783.feather, data/proofread_root_ids_783.npy): get them from https://zenodo.org/records/10676866 into data/"

[ $fail -eq 0 ] || { echo "FAIL $fail fetches"; exit 1; }
