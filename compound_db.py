#!/usr/bin/env python3
"""
compound_db.py
DuckDB-backed Embedded Chemoinformatics Registry & Morgan Fingerprint Vector Search
for the Drosophila Whole-Brain Connectome.
Indexes chemical structures, molecular formulas, CAS IDs, biophysical targets,
and maps Tanimoto molecular similarity to predicted connectome transmitter modulation.
"""

import sys
import os
import json
import argparse
import duckdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, DataStructs

DB_PATH = "compounds/compounds.duckdb"
PARQUET_PATH = "compounds/registry.parquet"
JSON_PATH = "compounds/registry.json"
PROFILES_DIR = "compounds/profiles"

# Modern RDKit Morgan Fingerprint Generator (Radius 2, 1024 bits)
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

# Curated reference seed library (Endogenous, Therapeutics, Insecticides, Neurotoxins)
SEED_COMPOUNDS = [
    # Endogenous Neurotransmitters
    {
        "slug": "acetylcholine",
        "name": "Acetylcholine",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "51-84-3",
        "smiles": "CC(=O)OCC[N+](C)(C)C",
        "primary_target": "nAChR / mAChR",
        "action_mode": "Endogenous Agonist",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "gaba",
        "name": "Gamma-Aminobutyric Acid (GABA)",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "56-12-2",
        "smiles": "C(CC(=O)O)CN",
        "primary_target": "GABA-A (Rdl) / GABA-B",
        "action_mode": "Endogenous Agonist (Inhibitory)",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.5, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "glutamate",
        "name": "L-Glutamic Acid",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "56-86-0",
        "smiles": "C(CC(=O)O)C(C(=O)O)N",
        "primary_target": "iGluR / GluClalpha",
        "action_mode": "Endogenous Agonist",
        "gains": {"ach": 1.0, "glu": 1.2, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "dopamine",
        "name": "Dopamine",
        "category": "Endogenous Neuromodulator",
        "cas_number": "51-61-6",
        "smiles": "c1cc(c(cc1CCN)O)O",
        "primary_target": "Dop1R1 / Dop1R2 / Dop2R",
        "action_mode": "Endogenous Agonist (Arousal)",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 1.5, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "serotonin",
        "name": "Serotonin (5-HT)",
        "category": "Endogenous Neuromodulator",
        "cas_number": "50-67-9",
        "smiles": "c1cc2c(cc1O)c(c[nH]2)CCN",
        "primary_target": "5-HT1A / 5-HT2A / 5-HT7",
        "action_mode": "Endogenous Agonist (State Control)",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 1.5, "oct": 0.3}
    },
    {
        "slug": "octopamine",
        "name": "Octopamine",
        "category": "Endogenous Neuromodulator",
        "cas_number": "104-14-3",
        "smiles": "c1cc(ccc1C(CN)O)O",
        "primary_target": "Oamb / Octbeta1R / Octbeta2R",
        "action_mode": "Endogenous Agonist (Fight-or-Flight)",
        "gains": {"ach": 1.2, "glu": 0.8, "gaba": -0.8, "dop": 0.5, "ser": 0.1, "oct": 3.0}
    },
    # Medical Countermeasures & Therapeutics
    {
        "slug": "atropine",
        "name": "Atropine",
        "category": "Medical Countermeasure",
        "cas_number": "51-55-8",
        "smiles": "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",
        "primary_target": "Muscarinic / Cholinergic Receptors",
        "action_mode": "Competitive Antagonist",
        "gains": {"ach": 0.3, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "pralidoxime",
        "name": "Pralidoxime (2-PAM)",
        "category": "Medical Countermeasure",
        "cas_number": "94-63-3",
        "smiles": "C[n+]1ccccc1C=NO",
        "primary_target": "Acetylcholinesterase (AChE)",
        "action_mode": "Enzymatic Oxime Reactivator",
        "gains": {"ach": 0.5, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "diazepam",
        "name": "Diazepam",
        "category": "Anticonvulsant / Sedative",
        "cas_number": "439-14-5",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "primary_target": "GABA-A Receptor Complex",
        "action_mode": "Positive Allosteric Modulator (PAM)",
        "gains": {"ach": 0.8, "glu": 0.7, "gaba": -2.5, "dop": 0.1, "ser": 0.2, "oct": 0.1}
    },
    {
        "slug": "midazolam",
        "name": "Midazolam",
        "category": "Anticonvulsant / Sedative",
        "cas_number": "59467-70-8",
        "smiles": "Cc1ncc2n1-c3ccc(Cl)cc3C(=Nc4ccccc4F)C2",
        "primary_target": "GABA-A Receptor Complex",
        "action_mode": "High-Affinity GABA-A PAM",
        "gains": {"ach": 0.8, "glu": 0.6, "gaba": -3.0, "dop": 0.1, "ser": 0.2, "oct": 0.1}
    },
    # Neuroactive Research Probes & Antagonists
    {
        "slug": "picrotoxinin",
        "name": "Picrotoxinin (PTX)",
        "category": "Convulsant Research Probe",
        "cas_number": "17617-45-7",
        "smiles": "CC(=C)C1C2CC34C15C(OC(=O)C56C3C2OC(=O)C6(C)O)O4",
        "primary_target": "GABA-A (Rdl) Chloride Channel",
        "action_mode": "Non-Competitive Channel Blocker",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": 0.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "bicuculline",
        "name": "Bicuculline",
        "category": "Convulsant Research Probe",
        "cas_number": "485-49-4",
        "smiles": "CN1CCC2=CC3=C(C=C2C1C4C5=CC6=C(C=C5C(=O)O4)OCO6)OCO3",
        "primary_target": "GABA-A Receptor",
        "action_mode": "Competitive Antagonist",
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -0.1, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "nicotine",
        "name": "Nicotine",
        "category": "Natural Alkaloid / Stimulant",
        "cas_number": "54-11-5",
        "smiles": "CN1CCCC1c2cccnc2",
        "primary_target": "Nicotinic AChR (nAChR)",
        "action_mode": "Agonist (Desensitizing)",
        "gains": {"ach": 2.2, "glu": 0.8, "gaba": -1.0, "dop": 0.8, "ser": 0.1, "oct": 0.5}
    },
    {
        "slug": "caffeine",
        "name": "Caffeine",
        "category": "CNS Stimulant",
        "cas_number": "58-08-2",
        "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "primary_target": "Adenosine Receptors / RyR",
        "action_mode": "Antagonist / Arousal Enhancer",
        "gains": {"ach": 1.2, "glu": 0.9, "gaba": -0.9, "dop": 0.5, "ser": 0.2, "oct": 1.2}
    },
    # Agricultural Agrochemicals
    {
        "slug": "ivermectin",
        "name": "Ivermectin (B1a)",
        "category": "Antiparasitic / Insecticide",
        "cas_number": "70288-86-7",
        "smiles": "CCC(C)C1CC(CC2(O1)CC3CC(O2)CC=C(C(C(C=CC=C4COC5C4(C(C=C(C5O)C)C(=O)O3)O)C)OC6CC(C(C(O6)C)OC7CC(C(C(O7)C)O)OC)OC)C)C",
        "primary_target": "GluClalpha (Chloride Channel)",
        "action_mode": "Allosteric Irreversible Activator",
        "gains": {"ach": 0.3, "glu": -3.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.1}
    },
    {
        "slug": "imidacloprid",
        "name": "Imidacloprid",
        "category": "Neonicotinoid Insecticide",
        "cas_number": "138261-41-3",
        "smiles": "c1cc(c(nc1)Cl)CN2CCNC2=N[N+](=O)[O-]",
        "primary_target": "Insect nAChR Subunits",
        "action_mode": "Irreversible Super-Agonist",
        "gains": {"ach": 3.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "amitraz",
        "name": "Amitraz",
        "category": "Formamidine Acaricide",
        "cas_number": "33089-61-1",
        "smiles": "Cc1ccc(cc1C)N=CN(C)C=Nc2c(C)cc(C)cc2",
        "primary_target": "Octopaminergic Oamb Receptor",
        "action_mode": "Agonist (Tremor Inducer)",
        "gains": {"ach": 1.2, "glu": 0.8, "gaba": -0.8, "dop": 0.5, "ser": 0.1, "oct": 4.5}
    },
    # Reference Acetylcholinesterase Probes
    {
        "slug": "paraoxon",
        "name": "Paraoxon-Ethyl",
        "category": "Organophosphate Probe",
        "cas_number": "311-45-5",
        "smiles": "CCOP(=O)(OCC)Oc1ccc(cc1)[N+](=O)[O-]",
        "primary_target": "Acetylcholinesterase (AChE)",
        "action_mode": "Irreversible Phosphorylation",
        "gains": {"ach": 6.0, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "dfp",
        "name": "Diisopropyl Fluorophosphate (DFP)",
        "category": "Organophosphate Reference",
        "cas_number": "55-91-4",
        "smiles": "CC(C)OP(=O)(F)OC(C)C",
        "primary_target": "Serine Hydrolases / AChE",
        "action_mode": "Irreversible Phosphorylation",
        "gains": {"ach": 5.5, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "malathion",
        "name": "Malathion",
        "category": "Organophosphate Insecticide",
        "cas_number": "121-75-5",
        "smiles": "CCOC(=O)CC(C(=O)OCC)SP(=S)(OC)OC",
        "primary_target": "AChE (via Malaoxon bioactivation)",
        "action_mode": "Phosphorylation",
        "gains": {"ach": 4.5, "glu": 0.9, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    }
]

def get_connection(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            compound_id VARCHAR PRIMARY KEY,
            slug VARCHAR,
            name VARCHAR,
            category VARCHAR,
            cas_number VARCHAR,
            formula VARCHAR,
            mw_g_mol FLOAT,
            canonical_smiles VARCHAR,
            inchikey VARCHAR,
            primary_target VARCHAR,
            action_mode VARCHAR,
            morgan_fp BLOB,
            g_ach FLOAT,
            g_glu FLOAT,
            g_gaba FLOAT,
            g_dop FLOAT,
            g_ser FLOAT,
            g_oct FLOAT
        );
    """)
    return con

def compute_chemo_descriptors(smiles: str):
    """Computes exact empirical formula, MW, InChIKey, and Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    can_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = float(Descriptors.MolWt(mol))
    inchikey = Chem.MolToInchiKey(mol)
    
    # 1024-bit Morgan Fingerprint
    fp = FP_GEN.GetFingerprint(mol)
    # Convert bit vector to binary text bytes
    fp_bytes = DataStructs.BitVectToBinaryText(fp)
    return can_smiles, formula, mw, inchikey, fp, fp_bytes

def init_registry():
    """Initializes the database from seed compounds and exports Parquet/JSON."""
    print("================ INITIALIZING COMPOUNDS DUCKDB REGISTRY ================")
    con = get_connection()
    os.makedirs(PROFILES_DIR, exist_ok=True)
    
    records_added = 0
    for idx, c in enumerate(SEED_COMPOUNDS):
        cmp_id = f"CMP-{idx+1:04d}"
        can_smiles, formula, mw, inchikey, fp, fp_bytes = compute_chemo_descriptors(c["smiles"])
        g = c["gains"]
        
        con.execute("""
            INSERT OR REPLACE INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cmp_id, c["slug"], c["name"], c["category"], c["cas_number"],
            formula, mw, can_smiles, inchikey, c["primary_target"], c["action_mode"],
            fp_bytes, g["ach"], g["glu"], g["gaba"], g["dop"], g["ser"], g["oct"]
        ))
        
        # Write individual profile JSON for screener
        profile = {
            "compound_id": cmp_id,
            "slug": c["slug"],
            "name": c["name"],
            "category": c["category"],
            "cas_number": c["cas_number"],
            "formula": formula,
            "mw_g_mol": mw,
            "canonical_smiles": can_smiles,
            "inchikey": inchikey,
            "primary_target": c["primary_target"],
            "action_mode": c["action_mode"],
            "gains": g
        }
        with open(f"{PROFILES_DIR}/{c['slug']}.json", "w") as f:
            json.dump(profile, f, indent=2)
            
        records_added += 1

    print(f"Indexed {records_added} compounds into {DB_PATH}")
    export_registry(con)
    con.close()

def export_registry(con=None):
    """Exports DuckDB table to columnar Parquet and canonical JSON."""
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True

    # Export to Parquet (excluding binary blob for clean interop)
    con.execute(f"""
        COPY (
            SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
                   canonical_smiles, inchikey, primary_target, action_mode,
                   g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
            FROM compounds
            ORDER BY compound_id
        ) TO '{PARQUET_PATH}' (FORMAT 'PARQUET');
    """)
    print(f"Exported columnar dataset to {PARQUET_PATH}")

    # Export to JSON
    rows = con.execute("""
        SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
               canonical_smiles, inchikey, primary_target, action_mode,
               g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
        FROM compounds
        ORDER BY compound_id
    """).fetchall()
    
    cols = [
        "compound_id", "slug", "name", "category", "cas_number", "formula", "mw_g_mol",
        "canonical_smiles", "inchikey", "primary_target", "action_mode",
        "g_ach", "g_glu", "g_gaba", "g_dop", "g_ser", "g_oct"
    ]
    json_records = []
    for r in rows:
        d = dict(zip(cols, r))
        d["gains"] = {
            "ach": d.pop("g_ach"), "glu": d.pop("g_glu"), "gaba": d.pop("g_gaba"),
            "dop": d.pop("g_dop"), "ser": d.pop("g_ser"), "oct": d.pop("g_oct")
        }
        json_records.append(d)
        
    with open(JSON_PATH, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"Exported canonical JSON index to {JSON_PATH}")

    if close_after:
        con.close()

def search_similarity(query_smiles: str, top_k: int = 5):
    """Searches registry by Morgan Fingerprint Tanimoto similarity."""
    q_mol = Chem.MolFromSmiles(query_smiles)
    if q_mol is None:
        print(f"Error: Invalid query SMILES: {query_smiles}")
        return []

    q_fp = FP_GEN.GetFingerprint(q_mol)
    q_formula = rdMolDescriptors.CalcMolFormula(q_mol)
    q_mw = Descriptors.MolWt(q_mol)

    con = get_connection()
    rows = con.execute("""
        SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
               canonical_smiles, primary_target, morgan_fp,
               g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
        FROM compounds
    """).fetchall()
    con.close()

    scored = []
    for r in rows:
        db_fp_bytes = r[9]
        db_fp = DataStructs.CreateFromBinaryText(db_fp_bytes)
        
        sim = DataStructs.TanimotoSimilarity(q_fp, db_fp)
        scored.append((sim, r))

    # Sort descending by Tanimoto similarity
    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    print("================ CHEMICAL SIMILARITY SEARCH (MORGAN FINGERPRINTS) ================")
    print(f"Query SMILES:  {query_smiles}")
    print(f"Query Formula: {q_formula} | MW: {q_mw:.1f} g/mol")
    print(f"Ranking Top {top_k} nearest chemical neighbors by Tanimoto coefficient:\n")

    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+")
    print("| Rank  | Tanim. | Compound Name            | Primary Target        | Formula     | CAS Number  |")
    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+")
    for rank, (sim, r) in enumerate(top_results, 1):
        name = r[2][:24]
        target = r[8][:21]
        formula = r[5][:11]
        cas = r[4]
        print(f"| #{rank:<4} | {sim*100:>5.1f}% | {name:<24} | {target:<21} | {formula:<11} | {cas:<11} |")
    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+")

    # Nearest neighbor prediction
    best_sim, best_r = top_results[0]
    print(f"\n>> PREDICTED CONNECTOME TRANSMITTER GAINS (via #{1} {best_r[2]} · {best_sim*100:.1f}% Match):")
    print(f"   ACh: {best_r[10]:.2f} | Glu: {best_r[11]:.2f} | GABA: {best_r[12]:.2f} | Dop: {best_r[13]:.2f} | Ser: {best_r[14]:.2f} | Oct: {best_r[15]:.2f}")
    return top_results

def list_compounds():
    """Lists all compounds currently registered in DuckDB."""
    con = get_connection()
    rows = con.execute("""
        SELECT compound_id, name, category, cas_number, formula, mw_g_mol, primary_target
        FROM compounds
        ORDER BY compound_id
    """).fetchall()
    con.close()

    print("================ REGISTERED COMPOUNDS IN DUCKDB ================")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+")
    print("| ID       | Name                              | Category                    | Formula     | CAS Number  |")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+")
    for r in rows:
        cid, name, cat, cas, form, mw, target = r
        print(f"| {cid:<8} | {name[:33]:<33} | {cat[:27]:<27} | {form:<11} | {cas:<11} |")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+")
    print(f"Total: {len(rows)} registered compounds.")

def main():
    parser = argparse.ArgumentParser(description="FlyBrain Chemoinformatics Database & Morgan Similarity Engine")
    subparsers = parser.add_subparsers(dest="cmd", help="Sub-commands")

    subparsers.add_parser("init", help="Initialize database from curated seed compounds")
    subparsers.add_parser("list", help="List registered compounds")
    subparsers.add_parser("export", help="Export DuckDB to Parquet and JSON")
    
    search_p = subparsers.add_parser("search", help="Search by SMILES Morgan similarity")
    search_p.add_argument("smiles", help="Target query SMILES string")
    search_p.add_argument("--top", type=int, default=5, help="Top N nearest neighbors (default: 5)")

    args = parser.parse_args()

    if args.cmd == "init" or not os.path.exists(DB_PATH):
        init_registry()
    elif args.cmd == "list":
        list_compounds()
    elif args.cmd == "export":
        export_registry()
    elif args.cmd == "search":
        search_similarity(args.smiles, top_k=args.top)
    else:
        list_compounds()

if __name__ == "__main__":
    main()
