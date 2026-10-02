#!/usr/bin/env python3
"""
biophysical_kinetics.py
Translates empirical pharmacological binding affinities (Ki, IC50, EC50 in nM)
and drug concentration into continuous synaptic conductance gains via Hill-Langmuir equations.

Eliminates arbitrary hand-tuned gain vectors.
Baseline neutral gain is 1.0 for all neurotransmitters.
"""

import math
from typing import Dict, Any, Optional

NT_CHANNELS = ["ach", "glu", "gaba", "dop", "ser", "oct"]

# Target to neurotransmitter channel mapping
TARGET_MAP = {
    # GABAergic targets
    "rdl": "gaba",
    "gaba_a": "gaba",
    "gaba_b": "gaba",
    "gaba receptor": "gaba",
    "gaba-a": "gaba",
    "chloride channel": "gaba",
    
    # Cholinergic targets
    "nachr": "ach",
    "machr": "ach",
    "ache": "ach",
    "acetylcholinesterase": "ach",
    "nicotinic": "ach",
    "muscarinic": "ach",
    
    # Glutamatergic targets
    "glucl": "glu",
    "mglur": "glu",
    "nmdar": "glu",
    "ampar": "glu",
    "vglut": "glu",
    
    # Dopaminergic targets
    "dop1r": "dop",
    "dop2r": "dop",
    "dat": "dop",
    "dopamine receptor": "dop",
    
    # Serotonergic targets
    "5-ht1": "ser",
    "5-ht2": "ser",
    "sert": "ser",
    "5-ht": "ser",
    "serotonin receptor": "ser",
    
    # Octopaminergic / Tyraminergic targets
    "oamb": "oct",
    "octr": "oct",
    "octopamine receptor": "oct",
    "octbeta": "oct",
}

def resolve_target_channel(target_str: Optional[str]) -> Optional[str]:
    """Maps target name to one of 6 neurotransmitter channels."""
    if not target_str:
        return None
    t = target_str.lower()
    for key, nt in TARGET_MAP.items():
        if key in t:
            return nt
    return None

def compute_hill_gain(
    affinity_nm: Optional[float],
    dose_nm: float,
    action_mode: str = "antagonist",
    hill_coeff: float = 1.0,
    emax: float = 2.0
) -> float:
    """
    Computes neurotransmitter conductance gain g from empirical affinity and dose.
    
    - Antagonist/Blocker: g = 1.0 / (1.0 + (dose / Ki)^n) -> drops from 1.0 to 0.0
    - Agonist: g = 1.0 + emax * (dose^n) / (dose^n + EC50^n) -> increases from 1.0 to 1.0+emax
    - Enzyme Inhibitor (e.g. AChE): g = 1.0 + emax * ((dose/IC50)^n) / (1 + (dose/IC50)^n) -> transmitter accumulation
    - Inactive / Null: returns 1.0
    """
    if dose_nm <= 0.0 or affinity_nm is None or affinity_nm <= 0.0:
        return 1.0
        
    mode = action_mode.lower()
    ratio = math.pow(dose_nm / affinity_nm, hill_coeff)
    
    if "antagonist" in mode or "blocker" in mode or "channel_blocker" in mode:
        # Conductance decreases towards 0
        g = 1.0 / (1.0 + ratio)
        return float(max(0.0, min(1.0, g)))
        
    elif "agonist" in mode:
        # Conductance increases up to 1.0 + emax
        g = 1.0 + (emax * ratio) / (1.0 + ratio)
        return float(g)
        
    elif "inhibitor" in mode or "ache" in mode or "nerve_agent" in mode:
        # E.g. AChE inhibition increases synaptic ACh presence
        g = 1.0 + (emax * ratio) / (1.0 + ratio)
        return float(g)
        
    elif "positive_allosteric_modulator" in mode or "pam" in mode:
        # Potentiation up to 1.0 + 0.8 * emax
        g = 1.0 + (0.8 * emax * ratio) / (1.0 + ratio)
        return float(g)
        
    # Default neutral
    return 1.0

def compute_compound_gains(
    compound: Dict[str, Any],
    dose_nm: Optional[float] = None,
    default_dose_multiple: float = 1.0
) -> Dict[str, float]:
    """
    Generates whole-brain gain vector for 6 neurotransmitters from compound metadata.
    
    If dose_nm is not specified, uses default_dose_multiple * affinity_val_nm.
    """
    gains = {nt: 1.0 for nt in NT_CHANNELS}
    
    target_str = compound.get("primary_target") or compound.get("bioassay_target")
    channel = resolve_target_channel(target_str)
    
    affinity = compound.get("affinity_val_nm")
    action_mode = compound.get("action_mode", "antagonist")
    
    if channel is not None and affinity is not None and affinity > 0.0:
        actual_dose = dose_nm if dose_nm is not None else (affinity * default_dose_multiple)
        gain_val = compute_hill_gain(
            affinity_nm=affinity,
            dose_nm=actual_dose,
            action_mode=action_mode
        )
        gains[channel] = round(gain_val, 4)
        
    return gains
