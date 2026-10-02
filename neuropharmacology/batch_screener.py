#!/usr/bin/env python3
"""
batch_screener.py
High-Throughput Whole-Brain Connectome Screening Engine: 100 Simulations.
Runs 100 distinct pharmacological conditions (dose titration and combination rescues)
on the AMD Radeon RX 7900 XTX 24GB via ROCm GPU kernel.
"""

import sys
import os
import json
import time
import subprocess
import duckdb
import pandas as pd
import numpy as np

PHARMA_BIN_DIR = "/home/<user>/Projects/mojo/fly-spmv"
PHARMA_BIN = "/home/<user>/Projects/mojo/fly-spmv/pharma_gpu"
DB_PATH = "compounds/compounds.duckdb"
OUTPUT_PARQUET = "data/batch_100_results.parquet"
OUTPUT_JSON = "data/batch_100_results.json"

def run_single_simulation(name: str, gains: dict, steps: int = 100) -> dict:
    """Executes one whole-brain GPU simulation via gpu-wait."""
    env_vars = [
        f"PHARMA_COMPOUND={name}",
        f"PHARMA_G_ACH={gains.get('ach', 1.0):.3f}",
        f"PHARMA_G_GLU={gains.get('glu', 1.0):.3f}",
        f"PHARMA_G_GABA={gains.get('gaba', 1.0):.3f}",
        f"PHARMA_G_DOP={gains.get('dop', 1.0):.3f}",
        f"PHARMA_G_SER={gains.get('ser', 1.0):.3f}",
        f"PHARMA_G_OCT={gains.get('oct', 1.0):.3f}",
        f"PHARMA_STEPS={steps}",
    ]
    
    cmd = [
        "gpu-wait", "run", "--vram", "4", "--timeout", "60", "--",
        "env"
    ] + env_vars + [PHARMA_BIN]
    
    t0 = time.time()
    res = subprocess.run(
        cmd,
        cwd=PHARMA_BIN_DIR,
        capture_output=True,
        text=True,
        check=True
    )
    wall_ms = (time.time() - t0) * 1000.0
    
    # Parse telemetry output
    telemetry = {
        "name": name,
        "steps": steps,
        "wall_ms": wall_ms
    }
    
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("PHARMA_RESULT_"):
            parts = line.split("=", 1)
            key = parts[0].replace("PHARMA_RESULT_", "").lower()
            val = parts[1].strip()
            try:
                parsed = float(val)
                telemetry[key] = int(parsed) if parsed.is_integer() else parsed
            except ValueError:
                telemetry[key] = val

    return telemetry

def build_100_experiment_manifest():
    """Builds exactly 100 experiment conditions: single-dose, high-dose, and rescue cocktails."""
    con = duckdb.connect(DB_PATH)
    rows = con.execute("""
        SELECT compound_id, slug, name, category, formula,
               g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
        FROM compounds
        ORDER BY compound_id
    """).fetchall()
    con.close()

    manifest = []
    
    # 1. 41 Compounds at Standard Dose (1.0x) -> 41 runs
    for r in rows:
        cid, slug, name, cat, form, ach, glu, gaba, dop, ser, oct_ = r
        manifest.append({
            "exp_id": len(manifest) + 1,
            "condition": "Standard Dose (1.0x)",
            "compound": name,
            "compound_id": cid,
            "dose_factor": 1.0,
            "gains": {"ach": ach, "glu": glu, "gaba": gaba, "dop": dop, "ser": ser, "oct": oct_}
        })
        
    # 2. 41 Compounds at High / Toxic Challenge Dose (2.5x gain deviation) -> 41 runs
    for r in rows:
        cid, slug, name, cat, form, ach, glu, gaba, dop, ser, oct_ = r
        # Scale deviation from neutral 1.0
        g_high = {
            "ach": 1.0 + (ach - 1.0) * 2.5,
            "glu": 1.0 + (glu - 1.0) * 2.5,
            "gaba": 1.0 + (gaba - 1.0) * 2.5,
            "dop": 1.0 + (dop - 1.0) * 2.5,
            "ser": 1.0 + (ser - 1.0) * 2.5,
            "oct": 1.0 + (oct_ - 1.0) * 2.5,
        }
        manifest.append({
            "exp_id": len(manifest) + 1,
            "condition": "High Dose Challenge (2.5x)",
            "compound": f"{name} [High]",
            "compound_id": cid,
            "dose_factor": 2.5,
            "gains": g_high
        })

    # 3. 18 Key Combination Rescues & Antidote Cocktails -> 18 runs (Total = 100)
    rescues = [
        # Sarin combinations
        ("Sarin Poisoning [Unmitigated]", {"ach": 5.0, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Sarin + Atropine (Muscarinic Block)", {"ach": 1.5, "glu": 0.9, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Sarin + 2-PAM (Reactivator)", {"ach": 2.2, "glu": 0.9, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Sarin + Dual (Atropine + 2-PAM)", {"ach": 1.05, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Sarin + Triple (Atropine + 2-PAM + Diazepam)", {"ach": 0.95, "glu": 0.8, "gaba": -1.3, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        
        # VX combinations
        ("VX Poisoning [Unmitigated]", {"ach": 7.5, "glu": 1.2, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("VX + Atropine", {"ach": 2.0, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("VX + 2-PAM", {"ach": 3.0, "glu": 1.1, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("VX + Dual Protocol", {"ach": 1.02, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("VX + Triple Protocol", {"ach": 0.98, "glu": 0.8, "gaba": -1.2, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        
        # Soman refractory aging case
        ("Soman Poisoning [AChE Aged]", {"ach": 9.0, "glu": 1.3, "gaba": -0.8, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Soman + 2-PAM (Aging Resistance Failure)", {"ach": 6.8, "glu": 1.2, "gaba": -0.8, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Soman + Triple Protocol (Diazepam Arrest)", {"ach": 1.15, "glu": 0.8, "gaba": -1.5, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        
        # Seizure and pesticide rescues
        ("Picrotoxin Seizure + Diazepam Rescue", {"ach": 1.0, "glu": 0.8, "gaba": -1.1, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Picrotoxin Seizure + Baclofen Rescue", {"ach": 0.9, "glu": 0.7, "gaba": -1.2, "dop": 0.2, "ser": 0.1, "oct": 0.3}),
        ("Imidacloprid Overdrive + Mecamylamine", {"ach": 0.9, "glu": 0.8, "gaba": -1.0, "dop": 0.3, "ser": 0.1, "oct": 0.3}),
        ("Fipronil Disinhibition + Muscimol", {"ach": 1.0, "glu": 0.8, "gaba": -1.4, "dop": 0.2, "ser": 0.1, "oct": 0.2}),
        ("Amphetamine Storm + Fluoxetine Blunting", {"ach": 1.1, "glu": 0.9, "gaba": -0.9, "dop": 1.8, "ser": 2.2, "oct": 1.4}),
    ]
    
    for r_name, r_gains in rescues:
        manifest.append({
            "exp_id": len(manifest) + 1,
            "condition": "Combination Rescue / Antidote",
            "compound": r_name,
            "compound_id": f"COCKTAIL-{len(manifest)+1:03d}",
            "dose_factor": 1.0,
            "gains": r_gains
        })

    assert len(manifest) == 100, f"Expected exactly 100 simulations, got {len(manifest)}"
    return manifest

def main():
    print("================================================================================")
    print(" FlyBrain High-Throughput Batch Screener: 100 Whole-Brain Simulations")
    print(" Target: 139,255 Neurons · AMD Radeon RX 7900 XTX 24GB (ROCm)")
    print("================================================================================\n")
    
    manifest = build_100_experiment_manifest()
    print(f"Generated test manifest: {len(manifest)} simulation conditions ready.\n")
    
    # 1. Run neutral baseline first for reference
    print("Calibrating neutral control baseline...")
    base = run_single_simulation("Baseline", {"ach": 1.0, "glu": 1.0, "gaba": 1.0, "dop": 1.0, "ser": 1.0, "oct": 1.0})
    b_act = base["mean_active"]
    b_dn = base["mean_dn"]
    b_sync = base["synchrony"]
    print(f"Baseline established: Active={b_act:.1f} | DN Motor={b_dn:.1f} | Sync={b_sync:.4f} | Latency={base['step_ms']:.2f} ms\n")
    
    results = []
    t_start = time.time()
    
    print(f"{'#':<4} | {'Condition':<28} | {'Active Spikes':<13} | {'Motor DN':<9} | {'Sync':<7} | {'GPU ms':<7} | {'Status'}")
    print("-" * 88)
    
    for idx, exp in enumerate(manifest, 1):
        name = exp["compound"]
        gains = exp["gains"]
        
        telemetry = run_single_simulation(name, gains, steps=100)
        
        act = telemetry["mean_active"]
        dn = telemetry["mean_dn"]
        sync = telemetry["synchrony"]
        step_ms = telemetry["step_ms"]
        
        delta_act = ((act - b_act) / b_act) * 100.0
        delta_dn = ((dn - b_dn) / b_dn) * 100.0
        delta_sync = ((sync - b_sync) / b_sync) * 100.0
        
        status = "NORMAL"
        if delta_act > 200 or delta_dn > 200:
            status = "SEIZURE"
        elif delta_act < -40 or delta_dn < -40:
            status = "PARALYSIS"
        elif abs(delta_act) < 15 and abs(delta_dn) < 15:
            status = "RESCUED"

        print(f"{idx:<4} | {name[:28]:<28} | {act:<13.1f} | {dn:<9.1f} | {sync:<7.3f} | {step_ms:<7.2f} | {status}")
        
        results.append({
            "exp_id": idx,
            "compound": name,
            "compound_id": exp["compound_id"],
            "condition": exp["condition"],
            "dose_factor": exp["dose_factor"],
            "mean_active": act,
            "mean_dn": dn,
            "synchrony": sync,
            "step_ms": step_ms,
            "delta_active_pct": delta_act,
            "delta_dn_pct": delta_dn,
            "delta_sync_pct": delta_sync,
            "status": status,
            "gains": gains
        })

    total_time = time.time() - t_start
    print("-" * 88)
    print(f"\nCompleted all {len(results)} whole-brain simulations in {total_time:.1f} seconds ({total_time/len(results):.2f}s per sim).")
    
    # Export datasets
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df = pd.DataFrame([{
        k: (json.dumps(v) if isinstance(v, dict) else v)
        for k, v in r.items()
    } for r in results])
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Exported batch results to Parquet: {OUTPUT_PARQUET}")
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported batch results to JSON: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
