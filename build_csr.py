#!/usr/bin/env python3
"""Build compressed sparse row (CSR) matrix of FlyWire connectome for Mojo/GPU SpMV."""

import numpy as np
import polars as pl
import os
import struct

def main():
    print("Building CSR matrix from FlyWire connections...")
    
    # 1. Load unique proofread neuron root IDs and create dense 0..N-1 mapping
    root_ids = np.load("data/proofread_root_ids_783.npy")
    num_neurons = len(root_ids)
    print(f"Total neurons: {num_neurons:,}")
    
    # Sort root_ids for fast binary search mapping
    root_to_idx = {rid: i for i, rid in enumerate(root_ids)}
    
    # 2. Load edge table
    df = pl.read_ipc("data/proofread_connections_783.feather")
    print(f"Total connections: {df.shape[0]:,}")
    
    # Sign weights by neurotransmitter if available:
    # Excitatory: ACh (+1), Inhibitory: GABA/Glut (-1), Modulatory: DA/Ser (+0.5)
    # Effective weight = syn_count * (ach_avg + 0.5*da_avg - gaba_avg - glut_avg)
    # Default to signed syn_count
    net_sign = pl.col("ach_avg") + 0.5 * pl.col("da_avg") - pl.col("gaba_avg") - pl.col("glut_avg")
    df = df.with_columns(
        (pl.col("syn_count").cast(pl.Float32) * pl.when(net_sign.abs() > 0.1).then(net_sign).otherwise(1.0).cast(pl.Float32)).alias("weight")
    )
    
    pre_ids = df["pre_pt_root_id"].to_numpy()
    post_ids = df["post_pt_root_id"].to_numpy()
    weights_np = df["weight"].to_numpy()
    
    # Filter only edges where both pre and post are in proofread set
    print("Mapping 64-bit IDs to 0..N-1 indices...")
    # Fast vectorized searchsorted mapping
    sort_order = np.argsort(root_ids)
    sorted_roots = root_ids[sort_order]
    
    # Map pre_ids
    pre_pos = np.searchsorted(sorted_roots, pre_ids)
    pre_valid = (pre_pos < num_neurons) & (sorted_roots[np.clip(pre_pos, 0, num_neurons - 1)] == pre_ids)
    
    # Map post_ids
    post_pos = np.searchsorted(sorted_roots, post_ids)
    post_valid = (post_pos < num_neurons) & (sorted_roots[np.clip(post_pos, 0, num_neurons - 1)] == post_ids)
    
    valid_mask = pre_valid & post_valid
    
    pre_idx = sort_order[pre_pos[valid_mask]].astype(np.int32)
    post_idx = sort_order[post_pos[valid_mask]].astype(np.int32)
    weights_np = weights_np[valid_mask]
    nnz = len(weights_np)
    print(f"Valid proofread edges (NNZ): {nnz:,}")
    
    # 3. Construct CSR: target (post) is row, source (pre) is col: y = W * x
    # Sort by post_idx (row)
    print("Sorting edges by row (post-synaptic target)...")
    order = np.lexsort((pre_idx, post_idx))
    post_idx = post_idx[order]
    pre_idx = pre_idx[order]
    weights_np = weights_np[order]
    
    # Compute row_ptr
    row_ptr = np.zeros(num_neurons + 1, dtype=np.int32)
    counts = np.bincount(post_idx, minlength=num_neurons)
    row_ptr[1:] = np.cumsum(counts)
    
    print(f"Row pointer max: {row_ptr[-1]:,}")
    print(f"Density: {nnz / (num_neurons * num_neurons) * 100:.4f}%")
    
    # 4. Save binary CSR file
    out_path = "data/fly_connectome_csr.bin"
    print(f"Saving binary CSR layout to {out_path}...")
    with open(out_path, "wb") as f:
        # Header: magic (4B), num_neurons (4B), nnz (4B)
        f.write(struct.pack("<III", 0x464C5943, num_neurons, nnz)) # 'FLYC'
        f.write(row_ptr.tobytes())
        f.write(pre_idx.tobytes())
        f.write(weights_np.tobytes())
        
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Successfully generated CSR binary: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
