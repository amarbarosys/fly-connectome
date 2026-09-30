#!/usr/bin/env python3
"""Inspect FlyWire connectome graph and cell annotations."""

import polars as pl
import numpy as np

def main():
    print("Loading FlyWire connectome graph...")
    # Load connectivity edge table
    df_conn = pl.read_ipc("data/proofread_connections_783.feather")
    print(f"Edge table loaded: {df_conn.shape[0]:,} directed connections")
    print(f"Schema: {df_conn.schema}")
    print("\nSample connections:")
    print(df_conn.head(5))

    # Load root IDs
    root_ids = np.load("data/proofread_root_ids_783.npy")
    print(f"\nProofread neurons (nodes): {len(root_ids):,}")

    # Summary statistics
    total_synapses = df_conn["syn_count"].sum() if "syn_count" in df_conn.columns else "N/A"
    print(f"Total synapses in graph: {total_synapses:,}")

    # Annotations
    annot_file = "data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
    try:
        df_annot = pl.read_csv(annot_file, separator="\t", truncate_ragged_lines=True)
        print(f"\nAnnotations loaded: {df_annot.shape[0]:,} neurons")
        print("Available annotation columns:")
        print(list(df_annot.columns)[:15])

        if "super_class" in df_annot.columns:
            print("\nNeurons by super class:")
            print(df_annot.group_by("super_class").len().sort("len", descending=True))
        
        if "top_nt" in df_annot.columns:
            print("\nPredicted neurotransmitter distribution:")
            print(df_annot.group_by("top_nt").len().sort("len", descending=True))
    except Exception as e:
        print(f"Could not load annotations: {e}")

if __name__ == "__main__":
    main()
