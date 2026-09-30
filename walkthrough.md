# Fly connectome perturbation walkthrough

## Status

This is a network perturbation demonstrator, not an independently validated pharmacological screener. Compound profiles apply illustrative transmitter gain changes. They do not implement dose-calibrated receptor pharmacology or establish biological phenotypes.

## Setup and run

The recorded run used the local Python environment and the Mojo engine in `/home/<user>/Projects/mojo/fly-spmv`:

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

## Independent validation status

No compatible reference endpoint was found in the checked local assets. FlyGym's bundled motion-capture clips describe limb kinematics, without matched drug exposure or neural counts. The local NWB file used by `sun_compass_verifier.py` supplies 42-ROI calcium traces for a separate ring-attractor demo, not FlyWire connectome activity under a matched compound. Neither supports a fair held-out test of the current aggregate metrics. No calibration/validation split, agreement score, or biological threshold is claimed.

To validate later, obtain a vehicle/baseline and a target-matched perturbation with an independent endpoint that can be mapped to the model. Pre-register the expected direction and endpoint, reserve distinct conditions for held-out evaluation, then report agreements, disagreements, and untestable predictions without tuning on held-out observations.

## Limitations

- No FlyWire-to-FCA neuron-level receptor map.
- Global gains do not represent receptor occupancy, dose, pharmacokinetics, or compound-specific cellular targets.
- Compound profiles are illustrative hypotheses and their magnitudes are not calibrated.
- The current active and descending counts are model telemetry, not observed behavior or physiology.
- No independent endpoint validates the pharmacological predictions; do not describe the output as clinical efficacy, toxicity prediction, seizure, paralysis, or tolerance.
- GPU throughput and step latency do not measure biological predictive performance.
