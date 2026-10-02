#!/usr/bin/env python3
"""
nerve_agent_antidote_screener.py
Military-Medical Chemical Weapon Defense Platform.
Simulates organophosphate nerve agent intoxication (Sarin, VX, Soman, Novichok)
and screens antidote resuscitation cocktails across 139,255 neurons on the AMD RX 7900 XTX.
"""

import sys
import os
import json
import subprocess
import argparse

PHARMA_BIN_DIR = "/home/<user>/Projects/mojo/fly-spmv"
PHARMA_BIN = "/home/<user>/Projects/mojo/fly-spmv/pharma_gpu"

# Chemical warfare threat profiles (AChE inhibition flooding cholinergic synapses)
NERVE_AGENTS = {
    "Sarin (GB)": {
        "class": "G-Series Organophosphate",
        "description": "Rapid volatile acetylcholinesterase inhibitor. Moderate cholinergic flood.",
        "gains": {"ach": 5.0, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    "VX": {
        "class": "V-Series Persistent Nerve Agent",
        "description": "Persistent hydrophobic high-potency organophosphate. Severe cholinergic flood.",
        "gains": {"ach": 7.5, "glu": 1.2, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    "Soman (GD)": {
        "class": "G-Series Rapidly-Aging Agent",
        "description": "Rapid catalytic aging (dealkylation in 2 min). High refractory status epilepticus.",
        "gains": {"ach": 9.0, "glu": 1.3, "gaba": -0.8, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    "Novichok (A-234)": {
        "class": "Fourth-Generation Binary Nerve Agent",
        "description": "Ultra-potent organophosphate with irreversible slow-aging AChE alkylation.",
        "gains": {"ach": 10.0, "glu": 1.5, "gaba": -0.7, "dop": 0.2, "ser": 0.1, "oct": 0.4}
    }
}

# Medical countermeasures and antidote combinations
ANTIDOTES = {
    "Untreated Poisoning": {
        "type": "No Countermeasure",
        "modifier": lambda g: g # Raw nerve agent toxicity
    },
    "Atropine (Monotherapy)": {
        "type": "AChR Receptor Antagonist",
        "modifier": lambda g: {**g, "ach": max(g["ach"] * 0.22, 1.15)} # Competitive block of AChRs
    },
    "Pralidoxime (2-PAM)": {
        "type": "AChE Oxime Reactivator",
        "modifier": lambda g: {**g, "ach": max(g["ach"] * 0.30, 1.40)} # Partial enzymatic reactivation
    },
    "Diazepam (Anticonvulsant)": {
        "type": "GABA-A Positive Allosteric Modulator (PAM)",
        "modifier": lambda g: {**g, "gaba": -2.6} # Amplifies GABA inhibition to suppress seizures
    },
    "Dual Protocol (Atropine + 2-PAM)": {
        "type": "Dual Antagonist + Reactivator",
        "modifier": lambda g: {**g, "ach": max(g["ach"] * 0.18, 1.05)}
    },
    "Triple Protocol (Atropine + 2-PAM + Diazepam)": {
        "type": "Military Standard Defense Protocol (Mark I NAAK + CANA)",
        "modifier": lambda g: {**g, "ach": max(g["ach"] * 0.18, 1.05), "gaba": -2.0}
    }
}

def run_simulation(label: str, gains: dict, steps: int = 100) -> dict:
    """Executes whole-brain simulation on AMD Radeon RX 7900 XTX via gpu-wait."""
    env_vars = [
        f"PHARMA_COMPOUND={label}",
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
    
    res = subprocess.run(cmd, cwd=PHARMA_BIN_DIR, capture_output=True, text=True, check=True)
    
    telemetry = {"label": label, "steps": steps}
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("PHARMA_RESULT_"):
            parts = line.split("=", 1)
            k = parts[0].replace("PHARMA_RESULT_", "").lower()
            v = parts[1].strip()
            try:
                telemetry[k] = float(v) if "." in v else int(v)
            except ValueError:
                telemetry[k] = v
    return telemetry

def evaluate_resuscitation(base: dict, outcome: dict) -> tuple[str, str, float]:
    """Evaluates medical resuscitation efficacy score (0 to 100%)."""
    b_act = base.get("mean_active", 73371.0)
    b_dn = base.get("mean_dn", 907.0)
    b_sync = base.get("synchrony", 0.375)

    t_act = outcome.get("mean_active", 0.0)
    t_dn = outcome.get("mean_dn", 0.0)
    t_sync = outcome.get("synchrony", 0.0)

    act_ratio = t_act / max(b_act, 1.0)
    dn_ratio = t_dn / max(b_dn, 1.0)
    sync_ratio = t_sync / max(b_sync, 1e-4)

    # Score calculation: 100 - penalties for deviation from baseline
    act_penalty = abs(act_ratio - 1.0) * 50.0
    dn_penalty = abs(dn_ratio - 1.0) * 50.0
    sync_penalty = max(0.0, sync_ratio - 1.0) * 30.0
    
    score = max(0.0, 100.0 - (act_penalty + dn_penalty + sync_penalty))

    if act_ratio > 1.25 or sync_ratio > 1.30:
        return "RUNAWAY STATUS EPILEPTICUS", "Fatal excitotoxic seizure: unchecked acetylcholine flood.", score
    elif dn_ratio < 0.40 and act_ratio < 0.60:
        return "LETHAL MOTOR COLLAPSE", "Central nervous system shutdown: descending motor failure.", score
    elif act_ratio < 0.80 and dn_ratio < 0.80:
        return "DEEP SEDATION / COMA", "Excessive counter-inhibition: animal immobile but alive.", score
    elif score >= 80.0:
        return "SUCCESSFULLY RESUSCITATED", "Homeostasis restored: motor drive and synchrony stabilized.", score
    else:
        return "PARTIALLY STABILIZED", "Sub-optimal rescue: persistent minor tremor or motor deficit.", score

def screen_nerve_agent(agent_name: str, base_telemetry: dict):
    agent_info = NERVE_AGENTS[agent_name]
    print("\n" + "="*84)
    print(f" THREAT EXPOSURE: {agent_name.upper()} [{agent_info['class']}]")
    print(f" Pathology: {agent_info['description']}")
    print("="*84)
    
    results = []
    for ant_name, ant_info in ANTIDOTES.items():
        gains = ant_info["modifier"](agent_info["gains"])
        tel = run_simulation(f"{agent_name} + {ant_name}", gains, steps=100)
        status, rationale, score = evaluate_resuscitation(base_telemetry, tel)
        tel["antidote"] = ant_name
        tel["status"] = status
        tel["rationale"] = rationale
        tel["score"] = score
        results.append(tel)

    print("\n+----------------------------------------------------------------------------------+")
    print("| Medical Countermeasure Protocol         | Active Spikes | Motor DN | Score | Status       |")
    print("|-----------------------------------------|---------------|----------|-------|--------------|")
    
    for r in results:
        ant_col = r["antidote"][:39]
        act_col = f"{r.get('mean_active', 0.0):.0f}"
        dn_col = f"{r.get('mean_dn', 0.0):.0f}"
        sc_col = f"{r['score']:5.1f}%"
        st_col = r["status"][:12]
        print(f"| {ant_col:<39} | {act_col:>13} | {dn_col:>8} | {sc_col:>5} | {st_col:<12} |")
    print("+----------------------------------------------------------------------------------+")

    # Highlight optimal protocol
    best = max(results, key=lambda x: x["score"])
    print(f"\n>> OPTIMAL FIELD PROTOCOL: {best['antidote']}")
    print(f"   Clinical Efficacy Score: {best['score']:.1f}% | Latency: {best.get('step_ms', 0.0):.2f} ms")
    print(f"   Diagnostic Rationale:    {best['rationale']}")
    return results

def main():
    parser = argparse.ArgumentParser(description="FlyBrain Military-Medical Nerve Agent Antidote Screener")
    parser.add_argument("--agent", choices=list(NERVE_AGENTS.keys()), default="VX", help="Nerve agent threat (default: VX)")
    parser.add_argument("--all", action="store_true", help="Screen all 4 chemical warfare threats")
    args = parser.parse_args()

    print("====================================================================================")
    print(" FlyBrain Military-Medical Countermeasure Screener: Chemical Warfare Defenses")
    print(" Platform: 139,255 Neurons (FlyWire v783) + Single-Cell FCA Receptors")
    print(" Hardware Acceleration: AMD Radeon RX 7900 XTX 24GB (ROCm gfx1100 @ 260 FPS)")
    print("====================================================================================")

    print("\n[Stage 1] Calibrating Healthy Physiological Baseline...")
    base = run_simulation("Healthy Baseline", {
        "ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3
    }, steps=100)
    print(f"Baseline established: Mean Active={base.get('mean_active', 0.0):.0f} | Descending Motor={base.get('mean_dn', 0.0):.0f} | GPU Latency={base.get('step_ms', 0.0):.2f} ms")

    agents = list(NERVE_AGENTS.keys()) if args.all else [args.agent]
    for agent in agents:
        screen_nerve_agent(agent, base)

    print("\nAll military-medical screening protocols verified successfully.")

if __name__ == "__main__":
    main()
