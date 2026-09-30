#!/usr/bin/env python3
"""Extract FlyWire neuron labels and audit whether FCA receptors can be mapped.

The current FCA file has barcode observations but no FlyWire root-ID crosswalk.
This script therefore emits no per-neuron receptor sensitivity matrix.
"""

import json
import numpy as np
import polars as pl
import anndata as ad

def main():
    root_ids_path = "data/proofread_root_ids_783.npy"
    annot_path = "data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
    h5ad_path = "data/fca_head.h5ad"

    print("[1/4] Loading connectome root IDs...")
    root_ids = np.load(root_ids_path)
    N = len(root_ids)
    print(f"Total connectome neurons: {N}")

    print("[2/4] Mapping neurotransmitter and super_class annotations...")
    df = pl.read_csv(
        annot_path,
        separator="\t",
        columns=["root_id", "super_class", "top_nt", "cell_class"]
    )
    
    # Map root_id to annotation record
    annot_map = {}
    for row in df.iter_rows(named=True):
        annot_map[row["root_id"]] = (row["top_nt"], row["super_class"], row["cell_class"])

    # NT codes:
    # 0: Unknown, 1: Acetylcholine, 2: Glutamate, 3: GABA, 4: Dopamine, 5: Serotonin, 6: Octopamine
    nt_code_map = {
        "acetylcholine": 1,
        "glutamate": 2,
        "gaba": 3,
        "dopamine": 4,
        "serotonin": 5,
        "octopamine": 6,
    }

    # Class codes:
    # 0: central, 1: sensory, 2: optic, 3: descending, 4: motor, 5: visual_centrifugal/other
    class_code_map = {
        "central": 0,
        "sensory": 1,
        "sensory_ascending": 1,
        "optic": 2,
        "visual_projection": 2,
        "descending": 3,
        "motor": 4,
    }

    nt_types = np.zeros(N, dtype=np.uint8)
    class_types = np.zeros(N, dtype=np.uint8)

    descending_indices = []

    for i, rid in enumerate(root_ids):
        if rid in annot_map:
            top_nt, s_class, c_class = annot_map[rid]
            if top_nt and top_nt in nt_code_map:
                nt_types[i] = nt_code_map[top_nt]
            if s_class and s_class in class_code_map:
                c_code = class_code_map[s_class]
                class_types[i] = c_code
                if c_code == 3:  # descending
                    descending_indices.append(i)
            else:
                class_types[i] = 0

    print(f"Annotated {np.count_nonzero(nt_types)} / {N} neurons with neurotransmitters:")
    for nt_name, code in nt_code_map.items():
        count = np.sum(nt_types == code)
        print(f"  {nt_name:<15}: {count:6d} neurons ({count/N*100:5.1f}%)")
    print(f"Identified {len(descending_indices)} descending motor command neurons (super_class == descending)")

    print("[3/4] Auditing FCA receptor mapping availability...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    receptor_genes = ["nAChRalpha1", "GluClalpha", "Rdl", "Dop1R1", "5-HT1A", "Oamb"]
    obs_columns = list(adata.obs.columns)
    candidate_crosswalk_columns = [
        c for c in obs_columns
        if any(term in c.lower() for term in ("root_id", "flywire", "segmentation", "neuron_id"))
    ]
    cns_count = int((adata.obs["afca_annotation_broad"] == "CNS neuron").sum())
    mapping_audit = {
        "mapping_mode": "no_map",
        "flywire_release": "783",
        "flywire_neurons": N,
        "flywire_nt_annotated": int(np.count_nonzero(nt_types)),
        "fca_file": h5ad_path,
        "fca_shape": [int(adata.n_obs), int(adata.n_vars)],
        "fca_obs_columns": obs_columns,
        "fca_observation_id_examples": [str(x) for x in adata.obs_names[:3]],
        "fca_cns_neurons": cns_count,
        "receptor_genes_present": {g: g in adata.var_names for g in receptor_genes},
        "candidate_flywire_crosswalk_columns": candidate_crosswalk_columns,
        "per_neuron_receptor_mapping_supported": False,
        "reason": "FCA observations are barcode-keyed and this file has no FlyWire root-ID/crosswalk field; population expression cannot be assigned to FlyWire neurons.",
        "receptor_sensitivity_matrix_written": False,
        "legacy_matrix_note": "An older neuron_receptor_sensitivities.bin may remain on disk; it is synthetic and must not be used as measured receptor data."
    }
    adata.file.close()

    print("[4/4] Writing binary artifacts...")
    nt_types.tofile("data/neuron_nt_types.bin")
    class_types.tofile("data/neuron_classes.bin")
    np.array(descending_indices, dtype=np.int32).tofile("data/descending_indices.bin")
    with open("data/receptor_mapping_audit.json", "w", encoding="utf-8") as f:
        json.dump(mapping_audit, f, indent=2)
        f.write("\n")

    print(f"Wrote data/neuron_nt_types.bin ({nt_types.nbytes} bytes)")
    print(f"Wrote data/neuron_classes.bin ({class_types.nbytes} bytes)")
    print(f"Wrote data/descending_indices.bin ({len(descending_indices)*4} bytes)")
    print("Wrote data/receptor_mapping_audit.json (mode=no_map)")
    print("FlyWire neurotransmitter extraction complete; no receptor map was inferred.")

if __name__ == "__main__":
    main()
