#!/usr/bin/env python3
"""
compound_db.py
DuckDB-backed Embedded Chemoinformatics Registry & Morgan Fingerprint Vector Search
for the Drosophila Whole-Brain Connectome.
Includes automated PubChem / ChEMBL API enrichment for bioassay Ki / IC50 / target affinities.
"""

import sys
import os
import json
import argparse
import duckdb
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, DataStructs

from chem_enricher import enrich_compound, fetch_pubchem_properties, fetch_chembl_bioactivity

DB_PATH = "compounds/compounds.duckdb"
PARQUET_PATH = "compounds/registry.parquet"
JSON_PATH = "compounds/registry.json"
PROFILES_DIR = "compounds/profiles"

# Modern RDKit Morgan Fingerprint Generator (Radius 2, 1024 bits)
FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

# 40 Curated Reference Compounds (Endogenous, Countermeasures, Probes, Insecticides)
SEED_COMPOUNDS = [
    # 1. Endogenous Neurotransmitters & Neuromodulators
    {
        "slug": "acetylcholine",
        "name": "Acetylcholine",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "51-84-3",
        "smiles": "CC(=O)OCC[N+](C)(C)C",
        "primary_target": "nAChR / mAChR",
        "action_mode": "Endogenous Agonist",
        "pubchem_cid": 187,
        "chembl_id": "CHEMBL667",
        "bioassay_target": "Neuronal acetylcholine receptor subunit alpha-7",
        "affinity_type": "EC50",
        "affinity_val_nm": 155000.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "gaba",
        "name": "Gamma-Aminobutyric Acid (GABA)",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "56-12-2",
        "smiles": "NCCCC(=O)O",
        "primary_target": "GABA-A (Rdl) / GABA-B",
        "action_mode": "Endogenous Agonist (Inhibitory)",
        "pubchem_cid": 119,
        "chembl_id": "CHEMBL96",
        "bioassay_target": "Gamma-aminobutyric acid receptor subunit alpha-1",
        "affinity_type": "EC50",
        "affinity_val_nm": 12000.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.5, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "glutamate",
        "name": "L-Glutamic Acid",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "56-86-0",
        "smiles": "C(CC(=O)O)C(C(=O)O)N",
        "primary_target": "iGluR / GluClalpha",
        "action_mode": "Endogenous Agonist",
        "pubchem_cid": 33032,
        "chembl_id": "CHEMBL421654",
        "bioassay_target": "Glutamate receptor ionotropic, NMDA 1",
        "affinity_type": "EC50",
        "affinity_val_nm": 2300.0,
        "gains": {"ach": 1.0, "glu": 1.2, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "dopamine",
        "name": "Dopamine",
        "category": "Endogenous Neuromodulator",
        "cas_number": "51-61-6",
        "smiles": "c1cc(c(cc1CCN)O)O",
        "primary_target": "Dop1R1 / Dop1R2 / Dop2R",
        "action_mode": "Endogenous Agonist (Arousal)",
        "pubchem_cid": 681,
        "chembl_id": "CHEMBL59",
        "bioassay_target": "Dopamine D1 receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 234.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 1.5, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "serotonin",
        "name": "Serotonin (5-HT)",
        "category": "Endogenous Neuromodulator",
        "cas_number": "50-67-9",
        "smiles": "c1cc2c(cc1O)c(c[nH]2)CCN",
        "primary_target": "5-HT1A / 5-HT2A / 5-HT7",
        "action_mode": "Endogenous Agonist (State Control)",
        "pubchem_cid": 5202,
        "chembl_id": "CHEMBL95",
        "bioassay_target": "Serotonin 1a (5-HT1a) receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 4.1,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 1.5, "oct": 0.3}
    },
    {
        "slug": "octopamine",
        "name": "Octopamine",
        "category": "Endogenous Neuromodulator",
        "cas_number": "104-14-3",
        "smiles": "c1cc(ccc1C(CN)O)O",
        "primary_target": "Oamb / Octbeta1R / Octbeta2R",
        "action_mode": "Endogenous Agonist (Fight-or-Flight)",
        "pubchem_cid": 4581,
        "chembl_id": "CHEMBL429",
        "bioassay_target": "Octopamine receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 48.0,
        "gains": {"ach": 1.2, "glu": 0.8, "gaba": -0.8, "dop": 0.5, "ser": 0.1, "oct": 3.0}
    },
    {
        "slug": "tyramine",
        "name": "Tyramine",
        "category": "Endogenous Neuromodulator",
        "cas_number": "51-67-2",
        "smiles": "c1cc(ccc1CCN)O",
        "primary_target": "TyrR (Tyramine Receptor)",
        "action_mode": "Trace Amine Agonist",
        "pubchem_cid": 5610,
        "chembl_id": "CHEMBL704",
        "bioassay_target": "Trace amine-associated receptor 1",
        "affinity_type": "EC50",
        "affinity_val_nm": 110.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -0.9, "dop": 0.3, "ser": 0.1, "oct": 1.8}
    },
    {
        "slug": "histamine",
        "name": "Histamine",
        "category": "Endogenous Neurotransmitter",
        "cas_number": "51-45-6",
        "smiles": "c1c(nc[nH]1)CCN",
        "primary_target": "HisCl1 / Ort (Histamine-gated chloride)",
        "action_mode": "Photoreceptor Inhibitory Transmitter",
        "pubchem_cid": 774,
        "chembl_id": "CHEMBL90",
        "bioassay_target": "Histamine H1 receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 450.0,
        "gains": {"ach": 0.9, "glu": 0.8, "gaba": -1.2, "dop": 0.1, "ser": 0.1, "oct": 0.2}
    },

    # 2. Medical Countermeasures & Antidotes
    {
        "slug": "atropine",
        "name": "Atropine",
        "category": "Medical Countermeasure",
        "cas_number": "51-55-8",
        "smiles": "CN1C2CCC1CC(C2)OC(=O)C(CO)c3ccccc3",
        "primary_target": "Muscarinic / Cholinergic Receptors",
        "action_mode": "Competitive Antagonist",
        "pubchem_cid": 174174,
        "chembl_id": "CHEMBL517712",
        "bioassay_target": "Muscarinic acetylcholine receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 0.45,
        "gains": {"ach": 0.3, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "pralidoxime",
        "name": "Pralidoxime (2-PAM)",
        "category": "Medical Countermeasure",
        "cas_number": "94-63-3",
        "smiles": "C[n+]1ccccc1C=NO",
        "primary_target": "Phosphorylated AChE Reactivator",
        "action_mode": "Oxime Reactivator",
        "pubchem_cid": 135398747,
        "chembl_id": "CHEMBL1420",
        "bioassay_target": "Acetylcholinesterase",
        "affinity_type": "IC50",
        "affinity_val_nm": 340000.0,
        "gains": {"ach": 0.5, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "obidoxime",
        "name": "Obidoxime",
        "category": "Medical Countermeasure",
        "cas_number": "114-90-9",
        "smiles": "C1=C[N+](=CC=C1C=NO)COC[N+]2=CC=C(C=C2)C=NO",
        "primary_target": "Bis-Pyridinium AChE Reactivator",
        "action_mode": "Catalytic Dephosphorylation",
        "pubchem_cid": 4543,
        "chembl_id": "CHEMBL3182163",
        "bioassay_target": "Acetylcholinesterase",
        "affinity_type": "IC50",
        "affinity_val_nm": 180000.0,
        "gains": {"ach": 0.4, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "diazepam",
        "name": "Diazepam",
        "category": "Anticonvulsant / Sedative",
        "cas_number": "439-14-5",
        "smiles": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
        "primary_target": "GABA-A Receptor Complex",
        "action_mode": "Positive Allosteric Modulator (PAM)",
        "pubchem_cid": 3016,
        "chembl_id": "CHEMBL12",
        "bioassay_target": "GABA-A receptor; anion channel",
        "affinity_type": "IC50",
        "affinity_val_nm": 12.0,
        "gains": {"ach": 0.8, "glu": 0.7, "gaba": -2.5, "dop": 0.1, "ser": 0.2, "oct": 0.1}
    },
    {
        "slug": "midazolam",
        "name": "Midazolam",
        "category": "Anticonvulsant / Sedative",
        "cas_number": "59467-70-8",
        "smiles": "CC1=NC=C2N1C3=C(C=C(C=C3)Cl)C(=NC2)C4=CC=CC=C4F",
        "primary_target": "GABA-A Receptor Complex",
        "action_mode": "Rapid-Onset Positive Allosteric Modulator",
        "pubchem_cid": 4192,
        "chembl_id": "CHEMBL644",
        "bioassay_target": "GABA-A receptor alpha-1 subunit",
        "affinity_type": "Ki",
        "affinity_val_nm": 4.8,
        "gains": {"ach": 0.7, "glu": 0.6, "gaba": -3.0, "dop": 0.1, "ser": 0.1, "oct": 0.1}
    },

    # 3. Convulsants & Receptor Probes
    {
        "slug": "picrotoxinin",
        "name": "Picrotoxinin (PTX)",
        "category": "Convulsant Research Probe",
        "cas_number": "17617-45-7",
        "smiles": "CC(=C)C1C2CC34C1(C5(C(=O)O3)C2(C(=O)OC5C4O)O)C",
        "primary_target": "GABA-A / Rdl Channel Pore",
        "action_mode": "Non-Competitive Channel Pore Blocker",
        "pubchem_cid": 442292,
        "chembl_id": "CHEMBL47244",
        "bioassay_target": "Glycine receptor subunit alpha-2",
        "affinity_type": "IC50",
        "affinity_val_nm": 2400.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": 0.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "bicuculline",
        "name": "Bicuculline",
        "category": "Convulsant Research Probe",
        "cas_number": "485-49-4",
        "smiles": "CN1CCC2=CC3=C(C=C2C1C4C5=C(C6=C(OCO6)C=C5)C(=O)O4)OCO3",
        "primary_target": "GABA-A Receptor",
        "action_mode": "Competitive Antagonist",
        "pubchem_cid": 10252,
        "chembl_id": "CHEMBL253164",
        "bioassay_target": "GABA-A receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 5000.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -0.2, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "baclofen",
        "name": "Baclofen",
        "category": "GABA-B Agonist / Spasmolytic",
        "cas_number": "1134-47-0",
        "smiles": "C1=CC(=CC=C1C(CC(=O)O)CN)Cl",
        "primary_target": "GABA-B Metabotropic Receptor",
        "action_mode": "Selective Agonist",
        "pubchem_cid": 2284,
        "chembl_id": "CHEMBL701",
        "bioassay_target": "GABA-B receptor",
        "affinity_type": "EC50",
        "affinity_val_nm": 1300.0,
        "gains": {"ach": 0.8, "glu": 0.6, "gaba": -2.2, "dop": 0.1, "ser": 0.1, "oct": 0.2}
    },
    {
        "slug": "muscimol",
        "name": "Muscimol",
        "category": "GABA-A Agonist / Sedative Probe",
        "cas_number": "2763-96-4",
        "smiles": "C1=C(ON=C1CN)O",
        "primary_target": "GABA-A Orthosteric Site",
        "action_mode": "Potent Full Agonist",
        "pubchem_cid": 4237,
        "chembl_id": "CHEMBL1310",
        "bioassay_target": "GABA-A receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 6.0,
        "gains": {"ach": 0.9, "glu": 0.7, "gaba": -3.5, "dop": 0.1, "ser": 0.1, "oct": 0.1}
    },
    {
        "slug": "kainic_acid",
        "name": "Kainic Acid",
        "category": "Excitotoxic Glutamate Probe",
        "cas_number": "487-79-6",
        "smiles": "CC(=C)C1C(C(=O)O)NC(C1CC(=O)O)C(=O)O",
        "primary_target": "Kainate / AMPA iGluR Subunits",
        "action_mode": "Excitotoxic Agonist",
        "pubchem_cid": 10255,
        "chembl_id": "CHEMBL279930",
        "bioassay_target": "Glutamate receptor ionotropic, kainate 1",
        "affinity_type": "Ki",
        "affinity_val_nm": 4.0,
        "gains": {"ach": 1.0, "glu": 4.5, "gaba": -1.0, "dop": 0.3, "ser": 0.1, "oct": 0.4}
    },
    {
        "slug": "nmda",
        "name": "N-Methyl-D-Aspartic Acid (NMDA)",
        "category": "Excitatory Amino Acid Probe",
        "cas_number": "6384-92-5",
        "smiles": "CN(CC(=O)O)C(C(=O)O)C",
        "primary_target": "NMDA Receptor Complex",
        "action_mode": "Selective Agonist",
        "pubchem_cid": 22880,
        "chembl_id": "CHEMBL286590",
        "bioassay_target": "NMDA receptor",
        "affinity_type": "EC50",
        "affinity_val_nm": 31000.0,
        "gains": {"ach": 1.0, "glu": 3.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },

    # 4. Cholinergic Agonists & Alkaloids
    {
        "slug": "nicotine",
        "name": "Nicotine",
        "category": "Natural Alkaloid / Stimulant",
        "cas_number": "54-11-5",
        "smiles": "CN1CCCC1c2cccnc2",
        "primary_target": "Insect / Mammalian nAChRs",
        "action_mode": "Partial / Full Agonist",
        "pubchem_cid": 89594,
        "chembl_id": "CHEMBL3",
        "bioassay_target": "Neuronal acetylcholine receptor; alpha4/beta2",
        "affinity_type": "Ki",
        "affinity_val_nm": 1.05,
        "gains": {"ach": 2.5, "glu": 1.0, "gaba": -1.0, "dop": 0.8, "ser": 0.2, "oct": 0.8}
    },
    {
        "slug": "carbachol",
        "name": "Carbachol",
        "category": "Cholinergic Research Agonist",
        "cas_number": "51-83-2",
        "smiles": "C[N+](C)(C)CCOC(=O)N",
        "primary_target": "nAChR and mAChR Subtypes",
        "action_mode": "Resistant to AChE Hydrolysis Agonist",
        "pubchem_cid": 5831,
        "chembl_id": "CHEMBL716",
        "bioassay_target": "Muscarinic acetylcholine receptor M1",
        "affinity_type": "EC50",
        "affinity_val_nm": 12000.0,
        "gains": {"ach": 2.2, "glu": 0.9, "gaba": -1.0, "dop": 0.3, "ser": 0.1, "oct": 0.4}
    },
    {
        "slug": "muscarine",
        "name": "Muscarine",
        "category": "Mushroom Alkaloid / Muscarinic Probe",
        "cas_number": "300-54-9",
        "smiles": "CC1C(CC(O1)C[N+](C)(C)C)O",
        "primary_target": "Muscarinic Acetylcholine Receptors",
        "action_mode": "Selective Agonist",
        "pubchem_cid": 9308,
        "chembl_id": "CHEMBL289728",
        "bioassay_target": "Muscarinic acetylcholine receptor M2",
        "affinity_type": "EC50",
        "affinity_val_nm": 420.0,
        "gains": {"ach": 1.8, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "epibatidine",
        "name": "Epibatidine",
        "category": "Frog Alkaloid / High-Affinity nAChR Probe",
        "cas_number": "140111-52-0",
        "smiles": "C1CC2CC1C(C2)c3cnc(cc3)Cl",
        "primary_target": "Nicotinic Acetylcholine Receptors",
        "action_mode": "Ultra-Potent Agonist",
        "pubchem_cid": 3279,
        "chembl_id": "CHEMBL421731",
        "bioassay_target": "Neuronal acetylcholine receptor; alpha4/beta2",
        "affinity_type": "Ki",
        "affinity_val_nm": 0.06,
        "gains": {"ach": 3.8, "glu": 1.1, "gaba": -1.0, "dop": 0.6, "ser": 0.2, "oct": 0.5}
    },
    {
        "slug": "mecamylamine",
        "name": "Mecamylamine",
        "category": "Ganglionic / Central nAChR Antagonist",
        "cas_number": "60-40-2",
        "smiles": "CC1(C2CCC(C2)(C1NC)C)C",
        "primary_target": "Nicotinic Channel Pore",
        "action_mode": "Non-Competitive Open Channel Blocker",
        "pubchem_cid": 4077,
        "chembl_id": "CHEMBL626",
        "bioassay_target": "Neuronal acetylcholine receptor; alpha3/beta4",
        "affinity_type": "IC50",
        "affinity_val_nm": 1200.0,
        "gains": {"ach": 0.2, "glu": 0.8, "gaba": -1.0, "dop": 0.1, "ser": 0.1, "oct": 0.2}
    },
    {
        "slug": "tubocurarine",
        "name": "d-Tubocurarine",
        "category": "Curare Alkaloid / nAChR Blocker",
        "cas_number": "57-95-4",
        "smiles": "CN1CCC2=CC3=C(C=C2C1CC4=CC=C(C=C4)OC5=C(C=CC(=C5)CC6C7=C(CCN6C)C=C(C(=C7)O)OC)O3)OC",
        "primary_target": "Neuromuscular & Ganglionic nAChRs",
        "action_mode": "Competitive Antagonist",
        "pubchem_cid": 6000,
        "chembl_id": "CHEMBL421453",
        "bioassay_target": "Muscle-type nicotinic acetylcholine receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 33.0,
        "gains": {"ach": 0.1, "glu": 0.8, "gaba": -1.0, "dop": 0.1, "ser": 0.1, "oct": 0.1}
    },

    # 5. Insecticides & Acaricides
    {
        "slug": "imidacloprid",
        "name": "Imidacloprid",
        "category": "Neonicotinoid Insecticide",
        "cas_number": "138261-41-3",
        "smiles": "c1cc(c(nc1)Cl)CN2CCNC2=N[N+](=O)[O-]",
        "primary_target": "Insect nAChR Subunits (Dalpha1/Dbeta1)",
        "action_mode": "Selective Agonist / Desensitizer",
        "pubchem_cid": 86289,
        "chembl_id": "CHEMBL1883196",
        "bioassay_target": "Insect nicotinic acetylcholine receptor",
        "affinity_type": "Ki",
        "affinity_val_nm": 1.9,
        "gains": {"ach": 3.0, "glu": 0.8, "gaba": -1.0, "dop": 0.4, "ser": 0.1, "oct": 0.5}
    },
    {
        "slug": "thiamethoxam",
        "name": "Thiamethoxam",
        "category": "Neonicotinoid Insecticide",
        "cas_number": "153719-23-4",
        "smiles": "CN1COCN(C1=N[N+](=O)[O-])Cc2csc(n2)Cl",
        "primary_target": "Insect nAChR Complex",
        "action_mode": "Neonicotinoid Pro-insecticide",
        "pubchem_cid": 54081,
        "chembl_id": "CHEMBL1883198",
        "bioassay_target": "Insect nicotinic acetylcholine receptor",
        "affinity_type": "IC50",
        "affinity_val_nm": 5.0,
        "gains": {"ach": 2.8, "glu": 0.8, "gaba": -1.0, "dop": 0.3, "ser": 0.1, "oct": 0.4}
    },
    {
        "slug": "fipronil",
        "name": "Fipronil",
        "category": "Phenylpyrazole Insecticide",
        "cas_number": "120068-37-3",
        "smiles": "C1=C(C=C(C(=C1Cl)N2C(=C(C(=N2)C#N)S(=O)C(F)(F)F)N)Cl)C(F)(F)F",
        "primary_target": "GABA-A / GluCl Anion Channels",
        "action_mode": "Non-Competitive Chloride Channel Blocker",
        "pubchem_cid": 3352,
        "chembl_id": "CHEMBL101326",
        "bioassay_target": "Gamma-aminobutyric acid receptor subunit beta-3",
        "affinity_type": "IC50",
        "affinity_val_nm": 28.0,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": 0.05, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "ivermectin",
        "name": "Ivermectin (B1a)",
        "category": "Antiparasitic / Insecticide",
        "cas_number": "70288-86-7",
        "smiles": "CCC(C)C1CC2CC(CC=C(C(=O)OC3CC4(CC(O3)C(=CC4)C(C)CC=CC=C(C(C(=O)O2)O1)C)OC5CC(C(C(O5)C)OC6CC(C(C(O6)C)O)OC)OC)C)O",
        "primary_target": "Glutamate-Gated Chloride (GluClalpha)",
        "action_mode": "Positive Allosteric Opener / Hyperpolarizer",
        "pubchem_cid": 6321424,
        "chembl_id": "CHEMBL1201103",
        "bioassay_target": "Glutamate-gated chloride channel",
        "affinity_type": "EC50",
        "affinity_val_nm": 150.0,
        "gains": {"ach": 0.8, "glu": -2.0, "gaba": -1.8, "dop": 0.1, "ser": 0.1, "oct": 0.2}
    },
    {
        "slug": "amitraz",
        "name": "Amitraz",
        "category": "Formamidine Acaricide",
        "cas_number": "33089-61-1",
        "smiles": "CC1=CC(=C(C=C1)N=CN(C)C=NC2=C(C=C(C=C2)C)C)C",
        "primary_target": "Octopaminergic Receptors (Octbeta2R)",
        "action_mode": "Agonist / Hyperexcitation",
        "pubchem_cid": 36324,
        "chembl_id": "CHEMBL293809",
        "bioassay_target": "Octopamine receptor",
        "affinity_type": "EC50",
        "affinity_val_nm": 320.0,
        "gains": {"ach": 1.1, "glu": 0.8, "gaba": -0.8, "dop": 0.5, "ser": 0.1, "oct": 4.5}
    },
    {
        "slug": "permethrin",
        "name": "Permethrin",
        "category": "Synthetic Pyrethroid Insecticide",
        "cas_number": "52645-53-1",
        "smiles": "CC1(C(C1C(=O)OCC2=CC(=CC=C2)OC3=CC=CC=C3)C=C(Cl)Cl)C",
        "primary_target": "Voltage-Gated Sodium Channels (Para)",
        "action_mode": "Slow Inactivation / Repetitive Spiking",
        "pubchem_cid": 40326,
        "chembl_id": "CHEMBL116674",
        "bioassay_target": "Sodium channel protein para",
        "affinity_type": "EC50",
        "affinity_val_nm": 85.0,
        "gains": {"ach": 1.5, "glu": 1.5, "gaba": -0.7, "dop": 0.4, "ser": 0.2, "oct": 0.5}
    },
    {
        "slug": "malathion",
        "name": "Malathion",
        "category": "Organophosphate Insecticide",
        "cas_number": "121-75-5",
        "smiles": "CCOC(=O)CC(SP(=S)(OC)OC)C(=O)OCC",
        "primary_target": "AChE (via Malaoxon bioactivation)",
        "action_mode": "Phosphorylation",
        "pubchem_cid": 4004,
        "chembl_id": "CHEMBL121",
        "bioassay_target": "Acetylcholinesterase",
        "affinity_type": "Ki",
        "affinity_val_nm": 1500.0,
        "gains": {"ach": 4.5, "glu": 0.9, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "paraoxon",
        "name": "Paraoxon-Ethyl",
        "category": "Organophosphate Probe",
        "cas_number": "311-45-5",
        "smiles": "CCOP(=O)(OCC)Oc1ccc(cc1)[N+](=O)[O-]",
        "primary_target": "Serine Hydrolases / AChE",
        "action_mode": "Covalent Inhibition",
        "pubchem_cid": 9295,
        "chembl_id": "CHEMBL190226",
        "bioassay_target": "Acetylcholinesterase",
        "affinity_type": "IC50",
        "affinity_val_nm": 18.0,
        "gains": {"ach": 6.5, "glu": 1.1, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },
    {
        "slug": "dfp",
        "name": "Diisopropyl Fluorophosphate (DFP)",
        "category": "Organophosphate Reference",
        "cas_number": "55-91-4",
        "smiles": "CC(C)OP(=O)(F)OC(C)C",
        "primary_target": "Serine Hydrolases / AChE",
        "action_mode": "Phosphorylation",
        "pubchem_cid": 5936,
        "chembl_id": "CHEMBL417463",
        "bioassay_target": "Cholinesterase",
        "affinity_type": "Ki",
        "affinity_val_nm": 45.0,
        "gains": {"ach": 5.5, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}
    },

    # 6. Stimulants, Modulators & Reuptake Inhibitors
    {
        "slug": "caffeine",
        "name": "Caffeine",
        "category": "CNS Stimulant",
        "cas_number": "58-08-2",
        "smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "primary_target": "Adenosine Receptors (AdoR) / Phosphodiesterase",
        "action_mode": "Competitive Antagonist",
        "pubchem_cid": 2519,
        "chembl_id": "CHEMBL113",
        "bioassay_target": "Adenosine receptor A1",
        "affinity_type": "Ki",
        "affinity_val_nm": 100000.0,
        "gains": {"ach": 1.1, "glu": 1.0, "gaba": -0.9, "dop": 0.8, "ser": 0.1, "oct": 1.2}
    },
    {
        "slug": "theophylline",
        "name": "Theophylline",
        "category": "Methylxanthine Stimulant",
        "cas_number": "58-55-9",
        "smiles": "CN1C2=C(C(=O)N(C1=O)C)NC=N2",
        "primary_target": "Adenosine Receptors / Phosphodiesterase",
        "action_mode": "Antagonist / Bronchodilator",
        "pubchem_cid": 2153,
        "chembl_id": "CHEMBL178",
        "bioassay_target": "Adenosine receptor A2a",
        "affinity_type": "Ki",
        "affinity_val_nm": 12000.0,
        "gains": {"ach": 1.1, "glu": 0.9, "gaba": -0.9, "dop": 0.6, "ser": 0.1, "oct": 1.0}
    },
    {
        "slug": "amphetamine",
        "name": "Amphetamine",
        "category": "CNS Stimulant / Transporter Substrate",
        "cas_number": "300-62-9",
        "smiles": "CC(CC1=CC=CC=C1)N",
        "primary_target": "DAT / VMAT2 / Trace Amine Receptors",
        "action_mode": "Dopamine & Octopamine Releaser",
        "pubchem_cid": 3007,
        "chembl_id": "CHEMBL39",
        "bioassay_target": "Sodium-dependent dopamine transporter",
        "affinity_type": "Ki",
        "affinity_val_nm": 34.0,
        "gains": {"ach": 1.1, "glu": 1.0, "gaba": -0.8, "dop": 3.2, "ser": 0.4, "oct": 2.5}
    },
    {
        "slug": "methylphenidate",
        "name": "Methylphenidate",
        "category": "Psychostimulant / Reuptake Inhibitor",
        "cas_number": "113-45-1",
        "smiles": "COC(=O)C(C1=CC=CC=C1)C2CCCCN2",
        "primary_target": "Dopamine Active Transporter (DAT)",
        "action_mode": "Reuptake Blocker",
        "pubchem_cid": 4158,
        "chembl_id": "CHEMBL549",
        "bioassay_target": "Sodium-dependent dopamine transporter",
        "affinity_type": "Ki",
        "affinity_val_nm": 60.0,
        "gains": {"ach": 1.0, "glu": 0.9, "gaba": -0.9, "dop": 2.8, "ser": 0.2, "oct": 1.5}
    },
    {
        "slug": "fluoxetine",
        "name": "Fluoxetine",
        "category": "Antidepressant / SSRI",
        "cas_number": "54910-89-3",
        "smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "primary_target": "Serotonin Transporter (SERT)",
        "action_mode": "Selective Serotonin Reuptake Inhibitor",
        "pubchem_cid": 3386,
        "chembl_id": "CHEMBL41",
        "bioassay_target": "Sodium-dependent serotonin transporter",
        "affinity_type": "Ki",
        "affinity_val_nm": 0.81,
        "gains": {"ach": 1.0, "glu": 0.8, "gaba": -1.0, "dop": 0.2, "ser": 3.5, "oct": 0.3}
    }
]

def get_connection(path: str = DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = duckdb.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            compound_id VARCHAR PRIMARY KEY,
            slug VARCHAR,
            name VARCHAR,
            category VARCHAR,
            cas_number VARCHAR,
            formula VARCHAR,
            mw_g_mol FLOAT,
            canonical_smiles VARCHAR,
            inchikey VARCHAR,
            primary_target VARCHAR,
            action_mode VARCHAR,
            pubchem_cid INTEGER,
            chembl_id VARCHAR,
            bioassay_target VARCHAR,
            affinity_type VARCHAR,
            affinity_val_nm FLOAT,
            iupac_name VARCHAR,
            morgan_fp BLOB,
            g_ach FLOAT,
            g_glu FLOAT,
            g_gaba FLOAT,
            g_dop FLOAT,
            g_ser FLOAT,
            g_oct FLOAT
        );
    """)
    return con

def compute_chemo_descriptors(smiles: str):
    """Computes exact empirical formula, MW, InChIKey, and Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    can_smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = float(Descriptors.MolWt(mol))
    inchikey = Chem.MolToInchiKey(mol)
    
    # 1024-bit Morgan Fingerprint
    fp = FP_GEN.GetFingerprint(mol)
    fp_bytes = DataStructs.BitVectToBinaryText(fp)
    return can_smiles, formula, mw, inchikey, fp, fp_bytes

def init_registry():
    """Initializes the database from seed compounds and exports Parquet/JSON."""
    print("================ INITIALIZING COMPOUNDS DUCKDB REGISTRY ================")
    con = get_connection()
    os.makedirs(PROFILES_DIR, exist_ok=True)
    
    records_added = 0
    for idx, c in enumerate(SEED_COMPOUNDS):
        cmp_id = f"CMP-{idx+1:04d}"
        can_smiles, formula, mw, inchikey, fp, fp_bytes = compute_chemo_descriptors(c["smiles"])
        g = c["gains"]
        
        cid = c.get("pubchem_cid")
        chembl = c.get("chembl_id")
        bio_target = c.get("bioassay_target")
        aff_type = c.get("affinity_type")
        aff_val = c.get("affinity_val_nm")
        iupac = c.get("iupac_name") or c["name"]
        
        con.execute("""
            INSERT OR REPLACE INTO compounds VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            cmp_id, c["slug"], c["name"], c["category"], c["cas_number"],
            formula, mw, can_smiles, inchikey, c["primary_target"], c["action_mode"],
            cid, chembl, bio_target, aff_type, aff_val, iupac,
            fp_bytes, g["ach"], g["glu"], g["gaba"], g["dop"], g["ser"], g["oct"]
        ))
        
        # Write individual profile JSON for screener
        profile = {
            "compound_id": cmp_id,
            "slug": c["slug"],
            "name": c["name"],
            "category": c["category"],
            "cas_number": c["cas_number"],
            "formula": formula,
            "mw_g_mol": mw,
            "canonical_smiles": can_smiles,
            "inchikey": inchikey,
            "primary_target": c["primary_target"],
            "action_mode": c["action_mode"],
            "pubchem_cid": cid,
            "chembl_id": chembl,
            "bioassay_target": bio_target,
            "affinity_type": aff_type,
            "affinity_val_nm": aff_val,
            "gains": g
        }
        with open(f"{PROFILES_DIR}/{c['slug']}.json", "w") as f:
            json.dump(profile, f, indent=2)
            
        records_added += 1

    print(f"Indexed {records_added} curated compounds into {DB_PATH}")
    export_registry(con)
    con.close()

def export_registry(con=None):
    """Exports DuckDB table to columnar Parquet and canonical JSON."""
    close_after = False
    if con is None:
        con = get_connection()
        close_after = True

    # Export to Parquet (excluding binary blob for clean interop)
    con.execute(f"""
        COPY (
            SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
                   canonical_smiles, inchikey, primary_target, action_mode,
                   pubchem_cid, chembl_id, bioassay_target, affinity_type, affinity_val_nm, iupac_name,
                   g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
            FROM compounds
            ORDER BY compound_id
        ) TO '{PARQUET_PATH}' (FORMAT 'PARQUET');
    """)
    print(f"Exported columnar dataset to {PARQUET_PATH}")

    # Export to JSON
    rows = con.execute("""
        SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
               canonical_smiles, inchikey, primary_target, action_mode,
               pubchem_cid, chembl_id, bioassay_target, affinity_type, affinity_val_nm, iupac_name,
               g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
        FROM compounds
        ORDER BY compound_id
    """).fetchall()
    
    cols = [
        "compound_id", "slug", "name", "category", "cas_number", "formula", "mw_g_mol",
        "canonical_smiles", "inchikey", "primary_target", "action_mode",
        "pubchem_cid", "chembl_id", "bioassay_target", "affinity_type", "affinity_val_nm", "iupac_name",
        "g_ach", "g_glu", "g_gaba", "g_dop", "g_ser", "g_oct"
    ]
    json_records = []
    for r in rows:
        d = dict(zip(cols, r))
        d["gains"] = {
            "ach": d.pop("g_ach"), "glu": d.pop("g_glu"), "gaba": d.pop("g_gaba"),
            "dop": d.pop("g_dop"), "ser": d.pop("g_ser"), "oct": d.pop("g_oct")
        }
        json_records.append(d)
        
    with open(JSON_PATH, "w") as f:
        json.dump(json_records, f, indent=2)
    print(f"Exported canonical JSON index to {JSON_PATH}")

    if close_after:
        con.close()

def add_compound_live(identifier: str):
    """
    Fetches compound live from PubChem & ChEMBL APIs, computes descriptors,
    infers nearest connectome gains, and persists to DuckDB.
    """
    print(f"================ QUERYING PUBCHEM & CHEMBL FOR: {identifier} ================")
    # 1. Fetch live enrichment
    enriched = enrich_compound(identifier)
    smiles = enriched.get("canonical_smiles")
    if not smiles:
        print(f"Error: Could not retrieve SMILES from PubChem for {identifier}")
        return False
        
    can_smiles, formula, mw, inchikey, fp, fp_bytes = compute_chemo_descriptors(smiles)
    
    # Connect to DB to check ID count and find nearest neighbor for gain prediction
    con = get_connection()
    count = con.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    cmp_id = f"CMP-{count+1:04d}"
    slug = identifier.lower().replace(" ", "_").replace("-", "_")
    name = identifier.title()
    category = "Live Enriched Query"
    primary_target = enriched.get("bioassay_target") or "Uncharacterized Target"
    action_mode = "Empirical Bioassay Match"
    
    # Compute similarity against existing compounds to infer gain vector
    rows = con.execute("SELECT compound_id, slug, name, morgan_fp, g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct FROM compounds").fetchall()
    scored = []
    for r in rows:
        db_fp = DataStructs.CreateFromBinaryText(r[3])
        sim = DataStructs.TanimotoSimilarity(fp, db_fp)
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    if scored:
        best_sim, best_r = scored[0]
        print(f"Mapped nearest structural neighbor: #{best_r[0]} {best_r[2]} ({best_sim*100:.1f}% Tanimoto similarity)")
        g = {
            "ach": best_r[4], "glu": best_r[5], "gaba": best_r[6],
            "dop": best_r[7], "ser": best_r[8], "oct": best_r[9]
        }
    else:
        g = {"ach": 1.0, "glu": 1.0, "gaba": -1.0, "dop": 0.2, "ser": 0.1, "oct": 0.3}

    con.execute("""
        INSERT OR REPLACE INTO compounds VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        cmp_id, slug, name, category, "N/A",
        formula, mw, can_smiles, inchikey, primary_target, action_mode,
        enriched.get("pubchem_cid"), enriched.get("chembl_id"),
        enriched.get("bioassay_target"), enriched.get("affinity_type"),
        enriched.get("affinity_val_nm"), enriched.get("iupac_name"),
        fp_bytes, g["ach"], g["glu"], g["gaba"], g["dop"], g["ser"], g["oct"]
    ))
    
    print(f"Persisted {name} ({cmp_id}) to DuckDB: CID={enriched.get('pubchem_cid')}, ChEMBL={enriched.get('chembl_id')}")
    export_registry(con)
    con.close()
    return True

def search_similarity(query_smiles: str, top_k: int = 5):
    """Searches registry by Morgan Fingerprint Tanimoto similarity."""
    q_mol = Chem.MolFromSmiles(query_smiles)
    if q_mol is None:
        print(f"Error: Invalid query SMILES: {query_smiles}")
        return []

    q_fp = FP_GEN.GetFingerprint(q_mol)
    q_formula = rdMolDescriptors.CalcMolFormula(q_mol)
    q_mw = Descriptors.MolWt(q_mol)

    con = get_connection()
    rows = con.execute("""
        SELECT compound_id, slug, name, category, cas_number, formula, mw_g_mol,
               canonical_smiles, primary_target, morgan_fp,
               pubchem_cid, chembl_id, affinity_type, affinity_val_nm,
               g_ach, g_glu, g_gaba, g_dop, g_ser, g_oct
        FROM compounds
    """).fetchall()
    con.close()

    scored = []
    for r in rows:
        db_fp_bytes = r[9]
        db_fp = DataStructs.CreateFromBinaryText(db_fp_bytes)
        
        sim = DataStructs.TanimotoSimilarity(q_fp, db_fp)
        scored.append((sim, r))

    # Sort descending by Tanimoto similarity
    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    print("================ CHEMICAL SIMILARITY SEARCH (MORGAN FINGERPRINTS) ================")
    print(f"Query SMILES:  {query_smiles}")
    print(f"Query Formula: {q_formula} | MW: {q_mw:.1f} g/mol")
    print(f"Ranking Top {top_k} nearest chemical neighbors by Tanimoto coefficient:\n")

    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+-------------+")
    print("| Rank  | Tanim. | Compound Name            | Primary Target        | Formula     | ChEMBL ID   | Affinity    |")
    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+-------------+")
    for rank, (sim, r) in enumerate(top_results, 1):
        name = r[2][:24]
        target = (r[8] or "N/A")[:21]
        formula = r[5][:11]
        chembl = (r[11] or "N/A")[:11]
        aff_type = r[12] or ""
        aff_val = f"{r[13]:.1f}nM" if r[13] is not None else "N/A"
        aff_str = f"{aff_type} {aff_val}"[:11]
        print(f"| #{rank:<4} | {sim*100:>5.1f}% | {name:<24} | {target:<21} | {formula:<11} | {chembl:<11} | {aff_str:<11} |")
    print("+-------+--------+--------------------------+-----------------------+-------------+-------------+-------------+")

    # Nearest neighbor prediction
    best_sim, best_r = top_results[0]
    print(f"\n>> PREDICTED CONNECTOME TRANSMITTER GAINS (via #{1} {best_r[2]} · {best_sim*100:.1f}% Match):")
    print(f"   ACh: {best_r[14]:.2f} | Glu: {best_r[15]:.2f} | GABA: {best_r[16]:.2f} | Dop: {best_r[17]:.2f} | Ser: {best_r[18]:.2f} | Oct: {best_r[19]:.2f}")
    return top_results

def list_compounds():
    """Lists all compounds currently registered in DuckDB."""
    con = get_connection()
    rows = con.execute("""
        SELECT compound_id, name, category, formula, pubchem_cid, chembl_id, primary_target, affinity_type, affinity_val_nm
        FROM compounds
        ORDER BY compound_id
    """).fetchall()
    con.close()

    print("================ REGISTERED COMPOUNDS IN DUCKDB (ENRICHED) ================")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+-------------+------------------+")
    print("| ID       | Name                              | Category                    | Formula     | CID         | ChEMBL ID   | Affinity         |")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+-------------+------------------+")
    for r in rows:
        cid, name, cat, form, pcid, chembl, target, aff_type, aff_val = r
        cid_str = str(pcid) if pcid else "N/A"
        chembl_str = chembl or "N/A"
        aff_str = f"{aff_type} {aff_val:.1f}nM" if aff_type and aff_val is not None else "N/A"
        print(f"| {cid:<8} | {name[:33]:<33} | {cat[:27]:<27} | {form:<11} | {cid_str:<11} | {chembl_str:<11} | {aff_str[:16]:<16} |")
    print("+----------+-----------------------------------+-----------------------------+-------------+-------------+-------------+------------------+")
    print(f"Total: {len(rows)} registered compounds.")

def main():
    parser = argparse.ArgumentParser(description="FlyBrain Chemoinformatics Database & Morgan Similarity Engine")
    subparsers = parser.add_subparsers(dest="cmd", help="Sub-commands")

    subparsers.add_parser("init", help="Initialize and seed database with 40 curated reference compounds")
    subparsers.add_parser("list", help="List registered compounds with bioassay data")
    subparsers.add_parser("export", help="Export DuckDB to Parquet and JSON")
    
    add_p = subparsers.add_parser("add", help="Live enrich and add a compound from PubChem/ChEMBL")
    add_p.add_argument("query", help="Compound name or identifier (e.g. 'fluoxetine', 'chlorpyrifos')")

    search_p = subparsers.add_parser("search", help="Search by SMILES Morgan similarity")
    search_p.add_argument("smiles", help="Target query SMILES string")
    search_p.add_argument("--top", type=int, default=5, help="Top N nearest neighbors (default: 5)")

    args = parser.parse_args()
    if args.cmd == "init":
        init_registry()
    elif args.cmd == "list":
        list_compounds()
    elif args.cmd == "export":
        export_registry()
    elif args.cmd == "add":
        add_compound_live(args.query)
    elif args.cmd == "search":
        search_similarity(args.smiles, top_k=args.top)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
