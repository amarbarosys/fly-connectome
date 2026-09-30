#!/usr/bin/env python3
"""Export 3D neuron coordinates and nerve connection hubs for Blender."""

import numpy as np
import polars as pl
import os

def main():
    annot_file = "data/annotations/flyconnectome-flywire_annotations-a83b277/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
    print("Reading annotations...")
    df = pl.read_csv(annot_file, separator="\t", truncate_ragged_lines=True)
    
    # Filter valid somas
    soma_df = df.filter(pl.col("soma_x").is_not_null())
    print(f"Total neurons with 3D somas: {soma_df.shape[0]:,}")
    
    sx = soma_df["soma_x"].to_numpy().astype(np.float64)
    sy = soma_df["soma_y"].to_numpy().astype(np.float64)
    sz = soma_df["soma_z"].to_numpy().astype(np.float64)
    super_classes = soma_df["super_class"].to_numpy()
    cell_types = soma_df["cell_type"].to_numpy()
    
    # FAFB center
    cx = (sx.min() + sx.max()) / 2.0
    cy = (sy.min() + sy.max()) / 2.0
    cz = (sz.min() + sz.max()) / 2.0
    
    # Convert to Blender millimeters:
    # X: Left-Right (voxel = 4nm = 0.000004 mm)
    # Y: Anterior-Posterior (FAFB Z * 40nm = 0.000040 mm)
    # Z: Dorsal-Ventral (FAFB -Y * 4nm = 0.000004 mm)
    bx = (sx - cx) * 0.000004
    by = -(sz - cz) * 0.000040
    bz = -(sy - cy) * 0.000004
    
    coords = np.column_stack([bx, by, bz]).astype(np.float32)
    
    class_map = {
        "optic": 0,
        "central": 1,
        "sensory": 2,
        "visual_projection": 3,
        "descending": 4,
        "motor": 5,
        "ascending": 6,
        "sensory_ascending": 7,
        "visual_centrifugal": 8,
        "endocrine": 9
    }
    
    class_ids = np.array([class_map.get(sc, 10) for sc in super_classes], dtype=np.int32)
    
    out_file = "data/fly_somas_blender.npz"
    np.savez_compressed(
        out_file,
        coords=coords,
        class_ids=class_ids,
        super_classes=super_classes,
        cell_types=cell_types
    )
    
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"Exported {coords.shape[0]:,} somas to {out_file} ({size_mb:.2f} MB)")
    print(f"Bounding box: X [{bx.min():.3f}, {bx.max():.3f}] mm, Y [{by.min():.3f}, {by.max():.3f}] mm, Z [{bz.min():.3f}, {bz.max():.3f}] mm")

if __name__ == "__main__":
    main()
