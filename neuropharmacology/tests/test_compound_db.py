"""Unit verification tests for Compound Database, Morgan Search, and PubChem/ChEMBL API Enricher."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb
from rdkit import Chem, DataStructs
from compound_db import get_connection, search_similarity, compute_chemo_descriptors
from chem_enricher import enrich_compound

def test_duckdb_schema_and_count():
    con = get_connection()
    count = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    # Check enriched columns exist
    cols = [c[1] for c in con.execute("PRAGMA table_info('compounds')").fetchall()]
    con.close()
    assert count >= 39, f"Expected at least 39 compounds, found {count}"
    assert "pubchem_cid" in cols
    assert "chembl_id" in cols
    assert "affinity_val_nm" in cols

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
    # Acetylcholine or Carbachol should match top
    assert top_row[1] in ("acetylcholine", "carbachol")
    assert top_sim > 0.50

def test_api_enricher():
    # Test live enrichment function
    res = enrich_compound("atropine")
    assert res.get("pubchem_cid") == 174174
    assert res.get("chembl_id") == "CHEMBL517712"
    assert res.get("affinity_type") == "Ki"
    assert res.get("affinity_val_nm") is not None

if __name__ == "__main__":
    test_duckdb_schema_and_count()
    test_chemo_descriptors()
    test_morgan_vector_similarity()
    test_api_enricher()
    print("ALL TESTS PASSED")
