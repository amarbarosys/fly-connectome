# Neuropharmacology Screener Update Plan

## Objective

Keep the current GPU simulation as a reproducible connectome perturbation prototype while making its receptor assumptions, outputs, and scientific claims auditable. Do not call a simulated perturbation a validated drug effect unless independent fly data supports it.

## Starting baseline (before this plan's first update)

- `fly-connectome` commit `9a0de20`: receptor extraction, compound profiles, and screening CLI.
- `fly-spmv` commit `f88b124`: Mojo GPU simulation engine.
- Reported hardware result: RX 7900 XTX / ROCm, roughly 3.8–3.9 ms per step; retain the existing panel as the comparison baseline.
- At plan start, the FCA receptor mean was not used to populate the emitted per-neuron matrix; most sensitivities were defaults. Drug profiles were hand-authored transmitter gains. The kernel multiplied already-signed CSR weights by those gains and by a receptor slot selected using presynaptic transmitter code. The CLI hard-coded 100 steps for compound runs, and phenotype names could overstate what the model establishes.

## Execution record: steps 1–4 completed (2026-09-30)

- Reproduced the pre-change baseline and four-profile panel on the RX 7900 XTX; all five runs completed. Hashes and outputs: `.work/neuropharmacology/reference-run-20260930-final/receipt.json`.
- Audited the FCA H5AD: barcode observations, no FlyWire identity crosswalk. Selected and implemented `no_map`; the extractor no longer emits synthetic per-neuron receptor densities. Details: `.work/neuropharmacology/step1-4-review.md` and `data/receptor_mapping_audit.json`.
- Verified CSR layout and found/fixed a sign inversion: CSR already stores negative GABA-associated weights, so the old neutral gain of -1 made them positive. Neutral gains are now 1.0 and receptor-sensitivity scaling is removed from the kernel.
- One-step CPU/GPU full-connectome counts match exactly (10,015 active; 333 descending). The updated five-condition panel passed; its receipt is `.work/neuropharmacology/no-map-run-final/receipt.json`.
- The compound profiles remain hand-authored transmitter-gain perturbations. Pharmacological mechanism and phenotype validation remain for steps 5–6.

## Execution record: step 5 completed (2026-09-30)

- The CLI passes the requested positive `--steps` value to the neutral control and every profile, and rejects absent, non-finite, negative, or mismatched required telemetry instead of substituting defaults.
- Scorecards report measured activity, descending activity, synchrony, latency, and control-relative changes. Synchrony is the standard deviation of per-step active-neuron counts divided by mean active-neuron count plus one. Output explicitly labels profile actions as unvalidated and makes no biological phenotype classification.
- No classification thresholds or supporting reference data exist; therefore the screener reports metric changes only. Biological validation remains step 6.
- Verification: `./.venv/bin/python -m unittest discover -s tests -v` passed all four focused tests, including a stubbed CLI run confirming seven steps reached both control and compound engine invocations. `py_compile`, `--help`, and `git diff --check` passed. No GPU workload was launched.

## Execution record: steps 6–7 advanced; biological validation remains unmet (2026-09-30)

- Audited local independent-data candidates. The bundled FlyGym assets are locomotion/motion-imitation clips without matched compound exposure or neural activity endpoints. `data/sub-13_ses-20201201_behavior_ophys.nwb` is used by the separate `sun_compass_verifier.py`, which builds an unrelated 42-unit ring attractor; its endpoint cannot validate this connectome-wide gain simulator. The FCA audit also confirms there is no FlyWire-to-FCA cell identity map.
- Downloaded and audited the Denecke et al. 2017 raw imidacloprid DGRP Wiggle Index file under `data/references/denecke_2017_imidacloprid_dgrp/`. A fixed analysis protocol was written before calculating outcomes or running the model, but after inspecting the file and labels, so it is not a prospective preregistration. The 0-minute values are within-well pre-exposure baselines, not a time-matched vehicle arm.
- On the held-out genotype subset (RAL number divisible by 5), median 60-minute/0-minute relative movement ratio was 0.716 at 25 ppm and 0.565 at 100 ppm; all 37 genotype ratios at each dose were below 1. Eight rows without parseable condition labels were excluded. Full tables and limitations are recorded in `data/references/denecke_2017_imidacloprid_dgrp/analysis_summary.json` and `comparison_report.md`.
- A 100-step GPU run of the unchanged imidacloprid gain profile increased mean descending-neuron activity by 283.60% vs neutral control, opposite the predeclared motor-suppression proxy direction. The profile also sets global GABA gain to -1.0, flipping already signed inhibitory edges and confounding attribution to imidacloprid's nAChR target. This is a diagnostic directional mismatch, not direct behavior-vs-neural validation: there is no dose-to-gain mapping, receptor-level action, vehicle control, or validated neural-to-locomotion mapping. No calibration or model tuning was performed. Step 6 remains **not biologically validated**. Receipt: `.work/neuropharmacology/denecke-imidacloprid-20260930/receipt.json`.
- Corrected the imidacloprid profile to preserve the existing illustrative ACh gain (3.0), set GABA and all other non-target gains to neutral 1.0, and label the entry an unvalidated global ACh-gain hypothesis. Synced `compounds/imidacloprid.json`, `compounds/profiles/imidacloprid.json`, the seed, DuckDB, and exported JSON/Parquet registry. This was based on the engine's signed-edge semantics and target specificity, without fitting to Denecke outcomes.
- A follow-up 100-step GPU run with the corrected profile still increased descending activity (+250.38%; 250.81 to 878.79), active count (+245.01%), and synchrony (+85.37%). It remains a simulator-only response: no receptor binding/desensitization or dose is represented. Receipt: `.work/neuropharmacology/denecke-corrected-profile-20260930/receipt.json`. The comparison protocol was not prospectively amended before this follow-up, so it is a sensitivity check, not new held-out validation.
- Step 7 walkthrough: `walkthrough.md` records setup, data provenance, exact run command, representative measured output and runtime receipt, and scientific limitations. Receipt: `.work/neuropharmacology/no-map-run-final/receipt.json` (100 steps, 3.16 ms reported step latency for neutral control, RX 7900 XTX run in the recorded GPU lane). These runtime numbers are performance evidence only.
- Release status: network perturbation demonstrator; not an independently validated pharmacological screener. Compound gains remain hypotheses. Step 7 walkthrough links the external dataset and both profile runs.

## Work plan

### 1. Freeze and reproduce the current reference run

Capture a clean baseline plus all four preset compounds with the current commits. Save the exact command, commit IDs, GPU/runtime identification, input hashes, stdout/stderr, and parsed telemetry under `.work/neuropharmacology/reference-run/`.

Acceptance:

- The run completes through `gpu-wait` on the intended GPU.
- Raw output and parsed scorecard agree for every metric.
- Reported latency is measured per run; no fallback constants are presented as measured values.
- Inputs and binaries are identified by hashes so later comparisons use the same baseline.

### 2. Correct receptor-data provenance and alignment

Audit the Fly Cell Atlas file’s gene identifiers, expression layer, cell annotations, and sampled-cell selection. Document that FCA cells do not have FlyWire `root_id` identities unless a supported crosswalk is found. Do not label a population average or class heuristic as per-neuron receptor density.

Implement one explicit supported mode:

- **Neuron-specific mode:** only if a defensible mapping from FlyWire neurons to receptor measurements exists; preserve coverage and uncertainty per neuron.
- **Cell-class mode:** summarize receptor expression by compatible cell class, record sample counts and missingness, and map only where class definitions have a defensible correspondence.
- **No-map mode:** if neither mapping is supported, omit receptor sensitivity from the simulation and label compound profiles as hypothetical transmitter perturbations.

Acceptance:

- Extraction emits provenance, gene IDs, expression source/layer, mapping method, coverage, and missing-value counts.
- A verifier checks vector lengths, finite/nonnegative values, known codes, and that reported coverage matches the data.
- No synthetic all-ones vector is described as measured receptor density.

### 3. Make synaptic and receptor semantics explicit

Document and verify the source and meaning of each edge weight, neurotransmitter label, receptor slot, and gain. Separate presynaptic transmitter identity from postsynaptic receptor identity. Apply a compound effect to a receptor only when its target and action are represented; avoid indexing receptor slots by transmitter code.

Acceptance:

- A small hand-checkable graph verifies edge direction, sign, transmitter assignment, receptor selection, and compound modulation.
- Unknown transmitter/receptor cases have explicit behavior and counts.
- Baseline produces finite, stable telemetry for a fixed seed/input; model constants and units are documented as assumptions where they are not experimentally measured.

### 4. Replace unsupported compound gains with auditable profiles

Represent each compound’s targets, action (agonist, antagonist, channel modulator), potency/dose assumptions, source citation, and confidence. Keep illustrative profiles available, but mark them `hypothesis` and do not present hand-selected gain values as measured pharmacology.

Acceptance:

- Each profile validates against a schema and reports missing target/action/dose evidence.
- Compound action changes the intended receptor path in a deterministic toy case.
- Baseline and vehicle controls are included; dose-response sweeps are enabled only when parameters have support.

### 5. Repair reporting and phenotype labels (implemented)

Remove fallback measured values and hard-coded latency. Honor the requested `--steps` value for both baseline and compound runs. Report raw metrics and deltas first. Define synchrony mathematically and distinguish population activity, motor-circuit activity, and synchrony.

Use claim-calibrated labels such as `increased activity`, `reduced descending activity`, or `increased synchrony`. Reserve terms such as `seizure`, `paralysis`, and `tolerated` for comparisons validated against an independent behavior or physiology reference.

Acceptance:

- Missing telemetry fails clearly rather than silently substituting defaults.
- A CLI smoke case confirms the same requested step count reaches baseline and compound runs.
- No classification thresholds are used until supporting reference data is available; output is limited to measured metric changes.

### 6. Validate against independent fly observations (partial directional check; biological validation unmet)

Choose a small reference panel with published or locally available neural/behavioral measurements, including vehicle/baseline and at least one perturbation with a matching model target. Use the FlyGym/behavior resources only where the measured endpoint and model output can be aligned. Keep calibration compounds separate from held-out validation compounds or conditions.

Acceptance:

- Pre-register the expected direction and endpoint for each reference condition before running it.
- Report agreement, disagreement, and untestable predictions; do not tune and score on the same observations.
- GPU speed is reported separately from biological predictive performance.

### 7. Deliver an auditable release candidate (walkthrough delivered)

Publish a concise `walkthrough.md` in the repository with setup, exact commands, hardware/runtime receipt, data provenance, limitations, and representative outputs. Keep generated logs and large reports under `.work/`; commit only code, schemas, compact fixtures, and documentation intended for reuse.

## Order and scope control

Do steps 1–3 before expanding the compound panel. Step 4 depends on step 2’s supported receptor mode. Step 5 can proceed alongside steps 2–3. Start validation only after outputs and mechanisms are interpretable. Preserve the current commits as the reference; do not overwrite their scorecard or claim that a code-only change establishes biological validity.

## Stop conditions

- If there is no defensible FlyWire-to-FCA cell mapping, stop calling receptor values neuron-specific and use the explicitly labeled no-map or cell-class mode.
- If compound dose/action evidence is unavailable, retain only hypothesis profiles and do not claim toxicity prediction.
- If no independent endpoint matches the simulated metrics, report the engine as a network perturbation demonstrator, not a validated pharmacological screener.
