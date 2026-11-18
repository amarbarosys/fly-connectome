# Fly connectome perturbation walkthrough

## Status

This is a network perturbation demonstrator, not an independently validated pharmacological screener. Compound profiles apply illustrative transmitter gain changes. They do not implement dose-calibrated receptor pharmacology or establish biological phenotypes.

## Setup and run

The commands below document the legacy single-compound Python workflow. For the current multi-condition batch build, follow the root [`README.md`](README.md): Rust `fly-screen` drives the resident Mojo GPU worker. A request for up to 50 conditions currently produces 48 rows so every compound retains all three dose levels.

The recorded legacy run used the local Python environment and the Mojo engine in `$FLY_SPMV_DIR` (default `../../mojo/fly-spmv` from the repo root):

```bash
./.venv/bin/python screen_compound.py --steps 100 --panel
```

The CLI sends `--steps` to both the neutral control and each selected compound. Each engine invocation is queued with `gpu-wait`. For a single profile:

```bash
./.venv/bin/python screen_compound.py --steps 100 compounds/picrotoxin.json
```

Required inputs are the CSR arrays, neuron transmitter labels, descending indices, and compiled `pharma_gpu` binary. The FCA receptor audit is in `data/receptor_mapping_audit.json`.

## Data provenance and model boundary

The graph is the FlyWire v783 proofread connectome. `build_csr.py` forms target-row/source-column CSR and signs each edge using available transmitter fractions. The current engine applies global transmitter gains to those already-signed weights. It is not a receptor-resolved or dose-response model.

The receptor extraction audit found 289,981 barcode-keyed Fly Cell Atlas observations and no FlyWire root-ID crosswalk. The supported mode is therefore `no_map`; receptor expression is not assigned to individual connectome neurons. Hand-authored profile gains are hypotheses, not measured compound actions.

## Representative recorded output

The run receipt at `.work/neuropharmacology/no-map-run-final/receipt.json` records a five-condition, 100-step GPU run. Representative neutral-control measurements:

| Metric | Recorded value |
| --- | ---: |
| Mean active neurons per step | 13,978.65 |
| Mean descending neurons per step | 250.81 |
| Population-count synchrony | 0.155929 |
| Reported step latency | 3.1627 ms |

The recorded target was an AMD Radeon RX 7900 XTX (gfx1100 / ROCm); the run was queued through `gpu-wait`. The receipt also stores engine and input hashes, commands, stdout/stderr paths, exit statuses, and per-profile telemetry. These are runtime/performance measurements, not biological validation results. Results describe that recorded engine and input snapshot; reproduce and save a new receipt before claiming a current build result.

Synchrony is the standard deviation of per-step active-neuron counts divided by the mean count plus one. Descending activity is an aggregate count over the engine's descending-index set; neither metric alone establishes a behavior such as walking, seizure, paralysis, or survival.

## Independent reference check

The Denecke et al. (2017) imidacloprid DGRP raw Wiggle Index CSV is stored in `data/references/denecke_2017_imidacloprid_dgrp/`, with provenance, checksum, analysis code, and a complete genotype-level summary. The source reports a larval motility assay at 25 and 100 ppm. The fixed held-out subset (RAL numeric ID divisible by 5; 37 genotypes per dose) had median 60-minute / pre-exposure relative movement ratios of 0.716 at 25 ppm and 0.565 at 100 ppm; all held-out genotype ratios were below 1 at both doses.

The original profile increased modeled descending-neuron activity by 283.60%; it also inverted signed GABA edges. I corrected it to an ACh-only global gain hypothesis (ACh 3.0; other transmitter gains neutral at 1.0). The corrected profile still increased descending activity by 250.38% (250.81 to 878.79), active-neuron count by 245.01%, and synchrony by 85.37%. Receipt: `.work/neuropharmacology/denecke-corrected-profile-20260930/receipt.json`. This follow-up is a sensitivity check, not an independent validation result.

This is **not a validated behavior prediction**. The source's 0-minute baseline is pre-exposure within-well, not a time-matched vehicle control. Larval motility is not the model's whole-connectome descending-neuron count, and the profile has no mapping from 25/100 ppm to transmitter gains. The corrected engine path still does not represent receptor binding or desensitization; its 3.0 ACh gain is illustrative. The profile correction followed signed-edge semantics and target specificity, not Denecke outcome values. The earlier local FlyGym motion-capture assets and unrelated 42-ROI NWB ring-attractor data still do not validate this screener.

Step 6's available directional comparison is complete and records a proxy disagreement: held-out larval movement decreased, while modeled descending activity increased. The study protocol was recorded before outcome summaries/model execution but after inspecting source labels, so it is not prospective preregistration. This does not constitute a matched endpoint validation or a valid prediction because no time-matched vehicle, ppm-to-gain calibration, or neural-to-locomotion mapping is available. Untestable with current evidence: vehicle-corrected chemical effects, calibrated dose response, neural activity correspondence, adult behavior, . The primary study describes the larval Wiggle Index exposure protocol and RMR endpoint ([Denecke et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5596007/)).

Biological validation remains open. Closing it requires independent fly observations with an appropriate time-matched vehicle, target- and dose-matched perturbation, and a defensible mapping from model output to measured behavior or physiology. Keep calibration separate from held-out evaluation and do not tune on held-out observations.

## Limitations

- No FlyWire-to-FCA neuron-level receptor map.
- Global gains do not represent receptor occupancy, dose, pharmacokinetics, or compound-specific cellular targets.
- Compound profiles are illustrative hypotheses and their magnitudes are not calibrated.
- The current active and descending counts are model telemetry, not observed behavior or physiology.
- No independent endpoint validates the pharmacological predictions; do not describe the output as clinical efficacy, toxicity prediction, seizure, paralysis, or tolerance.
- GPU throughput and step latency do not measure biological predictive performance.
