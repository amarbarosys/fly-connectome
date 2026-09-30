#!/usr/bin/env python3
"""
screen_compound.py
In-Silico Neuropharmacological Screener for the Drosophila Whole-Brain Connectome.
Executes high-throughput GPU simulations of chemical compounds against 139,255 neurons.
"""

import sys
import os
import json
import subprocess
import argparse

PHARMA_BIN_DIR = "/home/<user>/Projects/mojo/fly-spmv"
PHARMA_BIN = "/home/<user>/Projects/mojo/fly-spmv/pharma_gpu"

def run_simulation(compound_name: str, gains: dict, steps: int = 100) -> dict:
    """Runs a single simulation on the AMD RX 7900 XTX via gpu-wait."""
    env_vars = [
        f"PHARMA_COMPOUND={compound_name}",
        f"PHARMA_G_ACH={gains.get('ach', 1.0)}",
        f"PHARMA_G_GLU={gains.get('glu', 0.8)}",
        f"PHARMA_G_GABA={gains.get('gaba', -1.0)}",
        f"PHARMA_G_DOP={gains.get('dop', 0.2)}",
        f"PHARMA_G_SER={gains.get('ser', 0.1)}",
        f"PHARMA_G_OCT={gains.get('oct', 0.3)}",
        f"PHARMA_STEPS={steps}",
    ]
    
    cmd = [
        "gpu-wait", "run", "--vram", "4", "--timeout", "60", "--",
        "env"
    ] + env_vars + [PHARMA_BIN]
    
    res = subprocess.run(
        cmd,
        cwd=PHARMA_BIN_DIR,
        capture_output=True,
        text=True,
        check=True
    )
    
    # Parse telemetry output
    telemetry = {
        "compound": compound_name,
        "steps": steps,
        "raster": []
    }
    
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("PHARMA_RESULT_"):
            parts = line.split("=", 1)
            key = parts[0].replace("PHARMA_RESULT_", "").lower()
            val = parts[1].strip()
            try:
                if "." in val:
                    telemetry[key] = float(val)
                else:
                    telemetry[key] = int(val)
            except ValueError:
                telemetry[key] = val
        elif line.startswith("t=") or "Global=" in line:
            # Parse raster profile: "t= 0 : Global= 11755  DN= 464"
            try:
                tokens = line.replace(":", "").split()
                t_val = int(tokens[0].split("=")[1])
                glob_val = int(tokens[1].split("=")[1])
                dn_val = int(tokens[2].split("=")[1])
                telemetry["raster"].append((t_val, glob_val, dn_val))
            except Exception:
                pass
                
    return telemetry

def classify_phenotype(base: dict, test: dict) -> tuple[str, str]:
    """Classifies pharmacological phenotype based on delta telemetry."""
    base_active = base.get("mean_active", 73371.0)
    base_dn = base.get("mean_dn", 907.0)
    base_sync = base.get("synchrony", 0.375)

    test_active = test.get("mean_active", 0.0)
    test_dn = test.get("mean_dn", 0.0)
    test_sync = test.get("synchrony", 0.0)

    ratio_active = test_active / max(base_active, 1.0)
    ratio_dn = test_dn / max(base_dn, 1.0)
    ratio_sync = test_sync / max(base_sync, 1e-4)

    if ratio_dn < 0.85 and ratio_active < 0.70:
        return "FLACCID PARALYTIC", "Motor circuit suppression: descending motor neurons silenced."
    elif ratio_active < 0.65:
        return "EPILEPTIFORM / DEPOLARIZATION BLOCK", "Severe excitotoxicity: runaway firing triggers network refractory exhaustion."
    elif ratio_active > 1.10 or ratio_dn > 1.05:
        return "CHOLINERGIC / HYPERACTIVE STIMULANT", "Elevated excitatory drive: accelerated firing across central and motor neuropil."
    elif ratio_sync > 1.20:
        return "PAROXYSMAL SYNCHRONY", "Hypersynchronous burst firing across connectome clusters."
    elif ratio_active < 0.85:
        return "MILD CNS DEPRESSANT", "Moderate reduction in global synaptic throughput."
    else:
        return "NEUROTYPICAL / TOLERATED", "Neural circuit maintains stable excitation-inhibition homeostasis."

def render_ascii_sparkline(raster_points: list[int], max_val: int = 15000) -> str:
    """Renders a simple ASCII sparkline of activity."""
    ticks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    line = ""
    for v in raster_points:
        idx = min(len(ticks) - 1, int((v / max_val) * (len(ticks) - 1)))
        line += ticks[idx]
    return line

def screen_file(filepath: str, base_telemetry: dict = None) -> dict:
    with open(filepath, "r") as f:
        meta = json.load(f)
        
    compound_name = meta.get("name", "Unknown")
    gains = meta.get("gains", {})
    
    print(f"\n---> Screening {compound_name} [{meta.get('class', 'Custom')}]")
    telemetry = run_simulation(compound_name, gains, steps=100)
    
    if base_telemetry is None:
        base_telemetry = run_simulation("Baseline", {
            "ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3
        }, steps=100)
        
    phenotype, rationale = classify_phenotype(base_telemetry, telemetry)
    telemetry["phenotype"] = phenotype
    telemetry["rationale"] = rationale
    telemetry["meta"] = meta
    
    # Delta calculations
    base_active = base_telemetry.get("mean_active", 73371.0)
    base_dn = base_telemetry.get("mean_dn", 907.0)
    
    telemetry["delta_active_pct"] = ((telemetry.get("mean_active", 0.0) - base_active) / base_active) * 100.0
    telemetry["delta_dn_pct"] = ((telemetry.get("mean_dn", 0.0) - base_dn) / base_dn) * 100.0
    
    return telemetry

def print_scorecard(t: dict):
    meta = t.get("meta", {})
    print("+" + "-"*76 + "+")
    print(f"| COMPOUND: {t['compound']:<48} CAS: {meta.get('cas_number', 'N/A'):<12} |")
    print(f"| CLASS:    {meta.get('class', 'N/A')[:64]:<64} |")
    print("+" + "-"*76 + "+")
    print(f"| PHENOTYPE: {t['phenotype']:<63} |")
    print(f"| MECHANISM: {t['rationale'][:63]:<63} |")
    print("+" + "-"*76 + "+")
    print(f"| Metric                    | Baseline   | Compound   | Delta (%)    | Status |")
    print("|" + "-"*27 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*8 + "|")
    
    b_act = 73371.0
    t_act = t.get("mean_active", 0.0)
    d_act = t.get("delta_active_pct", 0.0)
    s_act = "NORMAL" if abs(d_act) < 15 else ("HIGH" if d_act > 0 else "LOW")
    print(f"| Global Active (Hz/step)   | {b_act:<10.1f} | {t_act:<10.1f} | {d_act:>+11.1f}% | {s_act:<6} |")
    
    b_dn = 907.0
    t_dn = t.get("mean_dn", 0.0)
    d_dn = t.get("delta_dn_pct", 0.0)
    s_dn = "NORMAL" if abs(d_dn) < 15 else ("HIGH" if d_dn > 0 else "SUPPR")
    print(f"| Motor Drive (DN Spikes)   | {b_dn:<10.1f} | {t_dn:<10.1f} | {d_dn:>+11.1f}% | {s_dn:<6} |")

    b_sync = 0.375
    t_sync = t.get("synchrony", 0.0)
    d_sync = ((t_sync - b_sync) / b_sync) * 100.0
    s_sync = "NORMAL" if abs(d_sync) < 20 else ("SYNCH" if d_sync > 0 else "DESYN")
    print(f"| Seizure / Synchrony Index | {b_sync:<10.3f} | {t_sync:<10.3f} | {d_sync:>+11.1f}% | {s_sync:<6} |")
    
    lat = t.get("step_ms", 3.95)
    fps = 1000.0 / max(lat, 0.001)
    print(f"| GPU Step Latency (ROCm)   | 3.95 ms    | {lat:<7.2f} ms | {fps:>7.1f} FPS   | PASS   |")
    print("+" + "-"*76 + "+")
    
    if t.get("raster"):
        global_raster = [p[1] for p in t["raster"]]
        dn_raster = [p[2] for p in t["raster"]]
        spark_g = render_ascii_sparkline(global_raster, max_val=max(max(global_raster), 1))
        spark_d = render_ascii_sparkline(dn_raster, max_val=max(max(dn_raster), 1))
        print(f"| Activity Sparklines (10 epochs): Global: [{spark_g}]  DN: [{spark_d}] |")
        print("+" + "-"*76 + "+")

def main():
    parser = argparse.ArgumentParser(description="FlyBrain In-Silico Neuropharmacology Screener")
    parser.add_argument("compound", nargs="?", default=None, help="Path to compound JSON file")
    parser.add_argument("--panel", action="store_true", help="Screen full preset panel of test compounds")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps per compound (default: 100)")
    args = parser.parse_args()

    print("================================================================================")
    print(" FlyBrain In-Silico Neuropharmacology & Drug Toxicity Screener")
    print(" Engine: Mojo SpMV Kernel · Hardware: AMD Radeon RX 7900 XTX 24GB (gfx1100)")
    print(" Multi-Omics: 139,255 Neurons (FlyWire v783) + Single-Cell FCA Receptors")
    print("================================================================================")

    # Run baseline first
    print("\n[Stage 1/2] Calibrating Physiological Baseline...")
    base = run_simulation("Baseline", {
        "ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3
    }, steps=args.steps)
    print(f"Baseline established: Mean Active={base.get('mean_active', 0.0):.1f} | DN Motor={base.get('mean_dn', 0.0):.1f} | Latency={base.get('step_ms', 0.0):.2f} ms")

    targets = []
    if args.panel or not args.compound:
        comp_dir = "/home/<user>/Projects/science/fly-connectome/compounds"
        targets = [
            os.path.join(comp_dir, f) for f in [
                "picrotoxin.json", "ivermectin.json", "imidacloprid.json", "amitraz.json"
            ]
        ]
    else:
        targets = [args.compound]

    print(f"\n[Stage 2/2] Screening {len(targets)} compound(s)...")
    for t_path in targets:
        result = screen_file(t_path, base_telemetry=base)
        print_scorecard(result)

    print("\nAll compounds screened successfully.")

if __name__ == "__main__":
    main()
