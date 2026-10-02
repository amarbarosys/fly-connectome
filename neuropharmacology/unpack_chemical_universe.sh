#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/home/<user>/Projects/science/fly-connectome/data/chemical_universe"
DOWNLOADS_DIR="/home/<user>/Projects/science/fly-connectome/data/downloads"

mkdir -p "${DEST_DIR}"

echo "=== UNPACKING CHEMICAL UNIVERSE ==="
echo "Target directory: ${DEST_DIR}"
echo "[1/2] Unpacking ChEMBL 37 SQLite database (30.48 GB)..."

tar -zxvf "${DOWNLOADS_DIR}/chembl_37_sqlite.tar.gz" \
    --strip-components=2 \
    -C "${DEST_DIR}" \
    "chembl_37/chembl_37_sqlite/chembl_37.db"

echo "[1/2] ChEMBL 37 SQLite unpacked: $(ls -lh "${DEST_DIR}/chembl_37.db" | awk '{print $5}')"

echo "[2/2] Decompressing PubChem 118M CID-SMILES (8.87 GB)..."
gzip -dc "${DOWNLOADS_DIR}/CID-SMILES.gz" > "${DEST_DIR}/CID-SMILES"

echo "[2/2] PubChem CID-SMILES unpacked: $(ls -lh "${DEST_DIR}/CID-SMILES" | awk '{print $5}')"

echo "=== ALL DATASETS UNPACKED AND AVAILABLE IN ${DEST_DIR} ==="
ls -lh "${DEST_DIR}"
