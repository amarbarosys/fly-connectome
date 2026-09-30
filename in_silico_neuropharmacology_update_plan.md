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

### 6. Validate against independent fly observations

Choose a small reference panel with published or locally available neural/behavioral measurements, including vehicle/baseline and at least one perturbation with a matching model target. Use the FlyGym/behavior resources only where the measured endpoint and model output can be aligned. Keep calibration compounds separate from held-out validation compounds or conditions.

Acceptance:

- Pre-register the expected direction and endpoint for each reference condition before running it.
- Report agreement, disagreement, and untestable predictions; do not tune and score on the same observations.
- GPU speed is reported separately from biological predictive performance.

### 7. Deliver an auditable release candidate

Publish a concise `walkthrough.md` in the repository with setup, exact commands, hardware/runtime receipt, data provenance, limitations, and representative outputs. Keep generated logs and large reports under `.work/`; commit only code, schemas, compact fixtures, and documentation intended for reuse.

## Order and scope control

Do steps 1–3 before expanding the compound panel. Step 4 depends on step 2’s supported receptor mode. Step 5 can proceed alongside steps 2–3. Start validation only after outputs and mechanisms are interpretable. Preserve the current commits as the reference; do not overwrite their scorecard or claim that a code-only change establishes biological validity.

## Stop conditions

- If there is no defensible FlyWire-to-FCA cell mapping, stop calling receptor values neuron-specific and use the explicitly labeled no-map or cell-class mode.
- If compound dose/action evidence is unavailable, retain only hypothesis profiles and do not claim toxicity prediction.
- If no independent endpoint matches the simulated metrics, report the engine as a network perturbation demonstrator, not a validated pharmacological screener.
