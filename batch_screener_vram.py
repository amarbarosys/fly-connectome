#!/usr/bin/env python3
"""
batch_screener_vram.py
Zero-disk RAM-resident high-throughput neuropharmacology screening engine.
Uses /dev/shm staged connectome CSR graphs, in-memory DuckDB compound index,
and mathematical Hill-Langmuir dose-response kinetics.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import duckdb
import pandas as pd
import numpy as np

from stage_ram_pipeline import stage_to_ram, verify_ram_stage
from biophysical_kinetics import compute_compound_gains

PHARMA_BIN_DIR = Path("/home/<user>/Projects/mojo/fly-spmv")
PHARMA_BIN = PHARMA_BIN_DIR / "pharma_gpu"
DB_PATH = Path("compounds/compounds.duckdb")
RAM_OUTPUT_PARQUET = Path("/dev/shm/fly-connectome/batch_ram_results.parquet")
PERSIST_OUTPUT_PARQUET = Path("data/batch_biophysical_results.parquet")

def load_in_memory_catalog() -> List[Dict[str, Any]]:
    """Loads compound catalog from disk into DuckDB :memory: table."""
    mem_con = duckdb.connect(":memory:")
    mem_con.execute(f"CREATE TABLE compounds AS SELECT * FROM read_parquet('compounds/registry.parquet');")
    df = mem_con.execute("SELECT * FROM compounds ORDER BY name").df()
    return df.to_dict(orient="records")

def run_simulation(name: str, gains: Dict[str, float], steps: int = 100) -> Dict[str, Any]:
    """Runs a single simulation on AMD RX 7900 XTX via gpu-wait."""
    env = os.environ.copy()
    env["PHARMA_COMPOUND"] = name
    env["PHARMA_G_ACH"] = f"{gains.get('ach', 1.0):.4f}"
    env["PHARMA_G_GLU"] = f"{gains.get('glu', 1.0):.4f}"
    env["PHARMA_G_GABA"] = f"{gains.get('gaba', 1.0):.4f}"
    env["PHARMA_G_DOP"] = f"{gains.get('dop', 1.0):.4f}"
    env["PHARMA_G_SER"] = f"{gains.get('ser', 1.0):.4f}"
    env["PHARMA_G_OCT"] = f"{gains.get('oct', 1.0):.4f}"
    env["PHARMA_STEPS"] = str(steps)
    env["FLY_DATA_DIR"] = "/dev/shm/fly-connectome/data/"

    cmd = [
        "gpu-wait", "run", "--vram", "4", "--timeout", "60", "--",
        str(PHARMA_BIN)
    ]
    
    t0 = time.time()
    res = subprocess.run(
        cmd,
        cwd=str(PHARMA_BIN_DIR),
        capture_output=True,
        text=True,
        env=env,
        check=True
    )
    wall_ms = (time.time() - t0) * 1000.0

    record = {
        "name": name,
        "steps": steps,
        "wall_ms": wall_ms,
        "g_ach": gains.get("ach", 1.0),
        "g_glu": gains.get("glu", 1.0),
        "g_gaba": gains.get("gaba", 1.0),
        "g_dop": gains.get("dop", 1.0),
        "g_ser": gains.get("ser", 1.0),
        "g_oct": gains.get("oct", 1.0),
    }

    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("PHARMA_RESULT_"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                k = parts[0].replace("PHARMA_RESULT_", "").lower().strip()
                v = parts[1].strip()
                try:
                    record[k] = float(v)
                except ValueError:
                    record[k] = v

    return record

def build_screening_manifest(catalog: List[Dict[str, Any]], target_count: int = 50) -> List[Dict[str, Any]]:
    """
    Builds dose-titration manifest using biophysical Hill kinetics:
    - 0.1x Ki/EC50 (sub-threshold)
    - 1.0x Ki/EC50 (half-maximal target saturation)
    - 10.0x Ki/EC50 (high challenge)
    - True negative controls (Water, Saline, Glucose)
    """
    manifest = []
    
    # Negative controls
    negatives = [
        {"name": "Water", "action_mode": "neutral", "affinity_val_nm": None, "primary_target": None},
        {"name": "Saline", "action_mode": "neutral", "affinity_val_nm": None, "primary_target": None},
        {"name": "D-Glucose", "action_mode": "neutral", "affinity_val_nm": None, "primary_target": None},
    ]
    for neg in negatives:
        gains = compute_compound_gains(neg)
        manifest.append({
            "name": f"{neg['name']} (Control)",
            "compound": neg["name"],
            "dose_level": "Control (0 nM)",
            "gains": gains
        })

    # Curated compounds with Hill dose sweeps
    for comp in catalog:
        c_name = comp["name"]
        aff = comp.get("affinity_val_nm")
        if not aff or np.isnan(aff):
            continue
            
        for multiplier, label in [(0.2, "Low (0.2x Ki)"), (1.0, "Standard (1.0x Ki)"), (5.0, "High (5.0x Ki)")]:
            dose_nm = aff * multiplier
            gains = compute_compound_gains(comp, dose_nm=dose_nm)
            manifest.append({
                "name": f"{c_name} [{label}]",
                "compound": c_name,
                "dose_level": label,
                "gains": gains
            })
            if len(manifest) >= target_count:
                break
        if len(manifest) >= target_count:
            break

    return manifest

def main():
    print("=================================================================")
    print(" Zero-Disk RAM-Resident Biophysical Screening Engine")
    print(" Target: AMD Radeon RX 7900 XTX 24GB via /dev/shm staging")
    print("=================================================================")

    # 1. Verify RAM staging
    print("[1/4] Checking /dev/shm connectome staging...")
    stage_to_ram(verbose=False)
    if not verify_ram_stage():
        print("[ERROR] RAM staging failed.")
        sys.exit(1)
        
    # 2. Load in-memory DuckDB catalog
    print("[2/4] Loading in-memory compound catalog...")
    catalog = load_in_memory_catalog()
    print(f"Catalog loaded: {len(catalog)} reference molecules in :memory: DuckDB.")

    # 3. Build manifest
    manifest = build_screening_manifest(catalog, target_count=50)
    print(f"[3/4] Generated biophysical Hill manifest: {len(manifest)} conditions.")

    # 4. Run simulations
    print(f"[4/4] Executing batch on GPU...")
    results = []
    t_start = time.time()
    
    for i, item in enumerate(manifest):
        name = item["name"]
        gains = item["gains"]
        rec = run_simulation(name, gains, steps=100)
        rec["compound"] = item["compound"]
        rec["dose_level"] = item["dose_level"]
        results.append(rec)
        print(f"  [{i+1:02d}/{len(manifest)}] {name:<35} | Act: {rec.get('active_pct', 0.0):5.1f}% | DN: {rec.get('dn_pct', 0.0):5.1f}% | Sync: {rec.get('synchrony', 0.0):.3f} | {rec['wall_ms']:.0f}ms")

    total_wall_s = time.time() - t_start
    print("=================================================================")
    print(f" Batch Complete: {len(results)} simulations in {total_wall_s:.1f}s ({total_wall_s/len(results):.2f}s/sim)")
    print("=================================================================")

    # Save to /dev/shm and persist
    df_res = pd.DataFrame(results)
    RAM_OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_parquet(RAM_OUTPUT_PARQUET)
    df_res.to_parquet(PERSIST_OUTPUT_PARQUET)
    print(f"Stream saved to {RAM_OUTPUT_PARQUET} and {PERSIST_OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
