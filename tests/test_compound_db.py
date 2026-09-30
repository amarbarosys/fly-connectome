"""Unit verification tests for Compound Database and Morgan Fingerprint Search."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb
from rdkit import Chem, DataStructs
from compound_db import get_connection, search_similarity, compute_chemo_descriptors

def test_duckdb_schema_and_count():
    con = get_connection()
    count = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    con.close()
    assert count >= 20, f"Expected at least 20 compounds, found {count}"

def test_chemo_descriptors():
    # Test Acetylcholine
    smiles = "CC(=O)OCC[N+](C)(C)C"
    can_smiles, formula, mw, inchikey, fp, fp_bytes = compute_chemo_descriptors(smiles)
    assert formula == "C7H16NO2+"
    assert abs(mw - 146.21) < 0.5
    assert len(fp_bytes) > 0

def test_morgan_vector_similarity():
    # Query with Carbachol (carbamoyl ester of choline)
    results = search_similarity("C[N+](C)(C)CCOC(=O)N", top_k=3)
    assert len(results) == 3
    top_sim, top_row = results[0]
    # Acetylcholine should be the top match
    assert top_row[1] == "acetylcholine"
    assert top_sim > 0.50

if __name__ == "__main__":
    test_duckdb_schema_and_count()
    test_chemo_descriptors()
    test_morgan_vector_similarity()
    print("ALL TESTS PASSED")
