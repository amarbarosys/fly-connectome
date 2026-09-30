#!/usr/bin/env python3
"""
chemical_universe.py
Unified query interface for unpacked ChEMBL 37 and PubChem 118M SMILES datasets.
Located in: data/chemical_universe/
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import duckdb

UNIVERSE_DIR = Path("/home/<user>/Projects/science/fly-connectome/data/chemical_universe")
CHEMBL_DB = UNIVERSE_DIR / "chembl_37.db"
SMILES_FILE = UNIVERSE_DIR / "CID-SMILES"

def is_ready() -> bool:
    """Checks if both datasets are unpacked and ready."""
    return CHEMBL_DB.exists() and SMILES_FILE.exists()

def get_chembl_connection() -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection attached to the unpacked ChEMBL 37 SQLite database."""
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found at {CHEMBL_DB}. Run unpack_chemical_universe.sh first.")
    
    con = duckdb.connect(":memory:")
    # Install and load sqlite extension for DuckDB if needed, or query sqlite directly
    con.execute(f"ATTACH '{CHEMBL_DB}' AS chembl (TYPE SQLITE, READ_ONLY);")
    return con

def query_chembl_targets(gene_name: str) -> List[Dict[str, Any]]:
    """Queries ChEMBL for assays matching a target gene or receptor name."""
    con = get_chembl_connection()
    query = """
    SELECT 
        m.chembl_id AS compound_chembl_id,
        m.pref_name,
        cs.canonical_smiles,
        t.pref_name AS target_name,
        t.organism,
        a.standard_type,
        a.standard_value,
        a.standard_units
    FROM chembl.target_dictionary t
    JOIN chembl.assays ass ON t.tid = ass.tid
    JOIN chembl.activities a ON ass.assay_id = a.assay_id
    JOIN chembl.molecule_dictionary m ON a.molregno = m.molregno
    LEFT JOIN chembl.compound_structures cs ON m.molregno = cs.molregno
    WHERE (LOWER(t.pref_name) LIKE ? OR LOWER(t.chembl_id) = ?)
      AND a.standard_type IN ('Ki', 'IC50', 'EC50')
      AND a.standard_value IS NOT NULL
    LIMIT 100;
    """
    pattern = f"%{gene_name.lower()}%"
    df = con.execute(query, [pattern, gene_name.lower()]).df()
    return df.to_dict(orient="records")

if __name__ == "__main__":
    print(f"Chemical Universe Status: Ready={is_ready()}")
    if CHEMBL_DB.exists():
        print(f"  - ChEMBL DB: {CHEMBL_DB} ({CHEMBL_DB.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
    if SMILES_FILE.exists():
        print(f"  - SMILES File: {SMILES_FILE} ({SMILES_FILE.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
