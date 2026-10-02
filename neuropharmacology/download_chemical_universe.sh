#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/home/<user>/Projects/science/fly-connectome/data/downloads"
mkdir -p "${DEST_DIR}"

CHEMBL_URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_sqlite.tar.gz"
PUBCHEM_URL="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"

echo "=== STARTING BACKGROUND DOWNLOAD: CHEMICAL UNIVERSE ==="
echo "Destination: ${DEST_DIR}"
echo "1. PubChem 118M SMILES (1.48 GB)"
echo "2. ChEMBL 37 Bioactive DB (5.40 GB)"
echo "Total download: 6.88 GB"
echo ""

# Download PubChem SMILES first (faster, 1.48 GB)
echo "[1/2] Downloading PubChem CID-SMILES.gz..."
curl -C - -L --retry 5 --retry-delay 5 \
     -o "${DEST_DIR}/CID-SMILES.gz" \
     "${PUBCHEM_URL}"

echo "[1/2] PubChem download complete."

# Download ChEMBL 37 SQLite tarball (5.40 GB) with 16 parallel connections
echo "[2/2] Downloading ChEMBL chembl_37_sqlite.tar.gz..."
aria2c -c -s 16 -x 16 -k 1M \
     -d "${DEST_DIR}" \
     -o "chembl_37_sqlite.tar.gz" \
     "${CHEMBL_URL}"

echo "[2/2] ChEMBL download complete."
echo "=== ALL DOWNLOADS COMPLETE (6.88 GB). UNPACKING DEFERRED. ==="
