#!/usr/bin/env python3
"""
screen_compound.py
In-Silico Neuropharmacological Screener for the Drosophila Whole-Brain Connectome.
Runs gain-perturbation simulations on the 139,255-neuron connectome.
"""

import sys
import os
import json
import subprocess
import argparse
import math

PHARMA_BIN_DIR = "/home/<user>/Projects/mojo/fly-spmv"
PHARMA_BIN = "/home/<user>/Projects/mojo/fly-spmv/pharma_gpu"

def run_simulation(compound_name: str, gains: dict, steps: int = 100) -> dict:
    """Run one simulation via gpu-wait and require complete engine telemetry."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    env_vars = [
        f"PHARMA_COMPOUND={compound_name}",
        f"PHARMA_G_ACH={gains.get('ach', 1.0)}",
        f"PHARMA_G_GLU={gains.get('glu', 1.0)}",
        f"PHARMA_G_GABA={gains.get('gaba', 1.0)}",
        f"PHARMA_G_DOP={gains.get('dop', 1.0)}",
        f"PHARMA_G_SER={gains.get('ser', 1.0)}",
        f"PHARMA_G_OCT={gains.get('oct', 1.0)}",
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
                parsed = float(val)
                telemetry[key] = int(parsed) if parsed.is_integer() else parsed
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
                
    require_telemetry(telemetry, requested_steps=steps)
    return telemetry

REQUIRED_METRICS = ("steps", "step_ms", "mean_active", "mean_dn", "synchrony")


def require_telemetry(telemetry: dict, requested_steps: int | None = None) -> None:
    missing = [key for key in REQUIRED_METRICS if key not in telemetry]
    if missing:
        raise RuntimeError(f"Simulator telemetry missing required metric(s): {', '.join(missing)}")
    for key in REQUIRED_METRICS:
        value = telemetry[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"Simulator telemetry {key} is not a finite number: {value!r}")
    if telemetry["steps"] <= 0 or not float(telemetry["steps"]).is_integer() or telemetry["step_ms"] <= 0:
        raise RuntimeError("Simulator telemetry steps and step_ms must be positive")
    if any(telemetry[key] < 0 for key in ("mean_active", "mean_dn", "synchrony")):
        raise RuntimeError("Simulator activity and synchrony metrics must be nonnegative")
    if requested_steps is not None and telemetry["steps"] != requested_steps:
        raise RuntimeError(f"Simulator ran {telemetry['steps']} steps; requested {requested_steps}")


def percent_delta(base_value: float, value: float, metric: str) -> float:
    if base_value == 0:
        raise ValueError(f"Cannot calculate percentage change: neutral-control {metric} is zero")
    return (value - base_value) / base_value * 100.0


def describe_perturbation(base: dict, test: dict) -> str:
    labels = (
        ("mean_active", "modeled active-neuron count"),
        ("mean_dn", "modeled descending-neuron activity"),
        ("synchrony", "model synchrony metric"),
    )
    parts = []
    for key, label in labels:
        delta = percent_delta(base[key], test[key], key)
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
        parts.append(f"{label} {direction} ({delta:+.1f}% vs neutral control)")
    return "; ".join(parts)

def render_ascii_sparkline(raster_points: list[int], max_val: int = 15000) -> str:
    """Renders a simple ASCII sparkline of activity."""
    ticks = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    line = ""
    for v in raster_points:
        idx = min(len(ticks) - 1, int((v / max_val) * (len(ticks) - 1)))
        line += ticks[idx]
    return line

def screen_file(filepath: str, base_telemetry: dict = None, steps: int = 100) -> dict:
    if base_telemetry is not None:
        require_telemetry(base_telemetry, requested_steps=steps)
    with open(filepath, "r") as f:
        meta = json.load(f)
        
    compound_name = meta.get("name", "Unknown")
    gains = meta.get("gains", {})
    
    print(f"\n---> Screening {compound_name} [gain perturbation profile]")
    telemetry = run_simulation(compound_name, gains, steps=steps)
    
    if base_telemetry is None:
        base_telemetry = run_simulation("Baseline", {
            "ach": 1.0, "glu": 1.0, "gaba": 1.0, "dop": 1.0, "ser": 1.0, "oct": 1.0
        }, steps=steps)
        
    require_telemetry(base_telemetry, requested_steps=steps)
    telemetry["interpretation"] = describe_perturbation(base_telemetry, telemetry)
    telemetry["meta"] = meta
    
    telemetry["delta_active_pct"] = percent_delta(base_telemetry["mean_active"], telemetry["mean_active"], "mean_active")
    telemetry["delta_dn_pct"] = percent_delta(base_telemetry["mean_dn"], telemetry["mean_dn"], "mean_dn")
    telemetry["delta_synchrony_pct"] = percent_delta(base_telemetry["synchrony"], telemetry["synchrony"], "synchrony")
    
    return telemetry

def print_scorecard(t: dict, base: dict):
    meta = t.get("meta", {})
    print("+" + "-"*76 + "+")
    print(f"| COMPOUND: {t['compound']:<48} CAS: {meta.get('cas_number', 'N/A'):<12} |")
    print(f"| DECLARED PROFILE (unvalidated): {meta.get('class', 'N/A')[:47]:<47} |")
    print("+" + "-"*76 + "+")
    print("| INTERPRETATION: modeled metric changes; not a biological phenotype claim")
    print(f"| {t['interpretation']}")
    print("+" + "-"*76 + "+")
    print("| Metric                      | Neutral control | Gain profile | Change       |")
    print("|-----------------------------|----------------:|-------------:|-------------:|")
    print(f"| Active neurons per step     | {base['mean_active']:>15.2f} | {t['mean_active']:>12.2f} | {t['delta_active_pct']:>+11.2f}% |")
    print(f"| Descending neurons per step | {base['mean_dn']:>15.2f} | {t['mean_dn']:>12.2f} | {t['delta_dn_pct']:>+11.2f}% |")
    print(f"| Synchrony metric            | {base['synchrony']:>15.6f} | {t['synchrony']:>12.6f} | {t['delta_synchrony_pct']:>+11.2f}% |")
    print(f"| Step latency (reported)     | {base['step_ms']:>12.3f} ms | {t['step_ms']:>9.3f} ms |              |")
    print("| Synchrony = population-count SD / (mean + 1); over simulation steps |")
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
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps for neutral control and each profile (default: 100)")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be a positive integer")

    print("================================================================================")
    print(" FlyWire Connectome Gain-Perturbation Prototype")
    print(" Engine: Mojo SpMV Kernel · Receptor mode: no neuron-level FCA map")
    print(" 139,255 FlyWire neurons; signed CSR weights; unvalidated gain profiles")
    print("================================================================================")

    # Run baseline first
    print("\n[Stage 1/2] Measuring neutral signed-connectome control...")
    base = run_simulation("Baseline", {
        "ach": 1.0, "glu": 1.0, "gaba": 1.0, "dop": 1.0, "ser": 1.0, "oct": 1.0
    }, steps=args.steps)
    print(f"Control measured: steps={base['steps']} | active={base['mean_active']:.1f} | descending={base['mean_dn']:.1f} | latency={base['step_ms']:.2f} ms")

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
        result = screen_file(t_path, base_telemetry=base, steps=args.steps)
        print_scorecard(result, base)

    print("\nAll compounds screened successfully.")

if __name__ == "__main__":
    main()
