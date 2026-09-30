"""
chem_enricher.py
PubChem and ChEMBL REST API automated scraper & bioassay data enricher.
Fetches bioassay Ki, IC50, molecular properties, and target bioactivity.
"""

import urllib.request
import urllib.parse
import json
import time
from typing import Optional, Dict, Any

USER_AGENT = "FlyBrainConnectomeChemo/1.0 (academic; connectome-pharmacology)"

def fetch_pubchem_properties(query: str, by: str = "name") -> Optional[Dict[str, Any]]:
    """
    Queries PubChem PUG REST API for compound properties.
    query: compound name or SMILES or CID
    by: 'name' or 'smiles' or 'cid'
    """
    encoded = urllib.parse.quote(str(query))
    if by == "smiles":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,IUPACName/JSON"
    elif by == "cid":
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,IUPACName/JSON"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,IUPACName/JSON"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                p = props[0]
                return {
                    "pubchem_cid": p.get("CID"),
                    "formula": p.get("MolecularFormula"),
                    "mw_g_mol": float(p.get("MolecularWeight", 0.0)),
                    "canonical_smiles": p.get("CanonicalSMILES") or p.get("ConnectivitySMILES"),
                    "inchikey": p.get("InChIKey"),
                    "iupac_name": p.get("IUPACName")
                }
    except Exception as e:
        # Fallback or silent log
        pass
    return None

def fetch_chembl_bioactivity(compound_name: str, chembl_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Queries ChEMBL REST API for molecule ID and target bioactivities (Ki, IC50, EC50).
    """
    try:
        if not chembl_id:
            # Query by molecule preferred name
            enc_name = urllib.parse.quote(compound_name.upper())
            mol_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?pref_name__iexact={enc_name}&limit=1"
            req = urllib.request.Request(mol_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as resp:
                mol_data = json.loads(resp.read().decode())
                mols = mol_data.get("molecules", [])
                if mols:
                    chembl_id = mols[0].get("molecule_chembl_id")
                else:
                    # Try text search if exact pref_name didn't hit
                    search_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={enc_name}&limit=1"
                    req_search = urllib.request.Request(search_url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req_search, timeout=8) as s_resp:
                        s_data = json.loads(s_resp.read().decode())
                        s_mols = s_data.get("molecules", [])
                        if s_mols:
                            chembl_id = s_mols[0].get("molecule_chembl_id")

        if not chembl_id:
            return None

        # Fetch activities for target affinities (Ki, IC50, EC50)
        act_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&standard_type__in=Ki,IC50,EC50&limit=5"
        req_act = urllib.request.Request(act_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req_act, timeout=8) as resp:
            act_data = json.loads(resp.read().decode())
            activities = act_data.get("activities", [])
            
            top_target = None
            top_val = None
            top_type = None
            top_units = None

            for a in activities:
                val = a.get("standard_value")
                if val is not None:
                    try:
                        fval = float(val)
                        top_target = a.get("target_pref_name") or a.get("assay_description")
                        top_val = fval
                        top_type = a.get("standard_type")
                        top_units = a.get("standard_units", "nM")
                        break
                    except ValueError:
                        continue

            return {
                "chembl_id": chembl_id,
                "bioassay_target": top_target,
                "affinity_type": top_type,
                "affinity_val_nm": top_val,
                "affinity_units": top_units
            }
    except Exception as e:
        pass
    return None

def enrich_compound(name: str, smiles: Optional[str] = None) -> Dict[str, Any]:
    """
    Enriches compound with both PubChem descriptors and ChEMBL bioactivities.
    """
    result = {
        "name": name,
        "pubchem_cid": None,
        "chembl_id": None,
        "bioassay_target": None,
        "affinity_type": None,
        "affinity_val_nm": None,
        "affinity_units": None,
        "iupac_name": None
    }

    # 1. PubChem
    pubchem_data = fetch_pubchem_properties(name, by="name")
    if not pubchem_data and smiles:
        pubchem_data = fetch_pubchem_properties(smiles, by="smiles")

    if pubchem_data:
        result.update(pubchem_data)

    # 2. ChEMBL
    chembl_data = fetch_chembl_bioactivity(name)
    if chembl_data:
        result.update(chembl_data)

    return result

if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "atropine"
    print(f"Fetching bioassay & chemical data for: {test_q}")
    enriched = enrich_compound(test_q)
    print(json.dumps(enriched, indent=2))
