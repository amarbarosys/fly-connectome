# Fly compound discovery protocol: imidacloprid stress and modifier screen

**Status:** computational comparison executed; wet-lab experiment not executed.  
**Purpose:** test whether candidate compounds modify a reproducible imidacloprid-associated fly phenotype, and determine whether the effect is consistent with a proposed mechanism. This is an early fly discovery screen, not evidence of human efficacy or safety.

## 1. Question and rationale

Can a candidate compound reduce imidacloprid-associated neuronal calcium disturbance and larval movement impairment in *Drosophila melanogaster* without independently impairing movement or survival?

Published work reports that 2.5 ppm imidacloprid produces sustained calcium elevation in nAChR-expressing larval neurons and reduces third-instar larval movement. The same study reports partial rescue of several effects by the antioxidant N-acetylcysteine amide (NACA), making NACA a positive control for a downstream oxidative-stress rescue, not a positive control for blocking nAChRs. See [the primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7568275/).

The connectome simulator currently applies global transmitter-gain hypotheses. It does not represent receptor occupancy, desensitization, internal exposure, calcium dynamics, oxidative stress, or larval movement. Therefore it cannot rank candidate compounds for this assay. Its role is limited to proposing circuit hypotheses after the compound target and model path are specified.

## 2. Preregister before collecting new data

Before unblinding or running candidate conditions, save a dated protocol copy with:

- fly strain/genotype, developmental stage, rearing conditions, sex where applicable, and exclusion criteria;
- compound identity, purity/source, solvent, concentration series, exposure route and duration;
- primary endpoint, effect direction, experimental unit, randomization/blocking, planned sample size and power rationale;
- statistical model, multiplicity correction, missing-data rules, and a minimum meaningful rescue effect;
- candidate IDs and a held-out confirmation set that will not be used for tuning.

The historical Denecke analysis in this repository is exploratory external evidence: its protocol was written after source-file inspection, its baseline is pre-exposure within-well rather than time-matched vehicle, and its held-out genotype split is not a new prospective experiment.

## 3. Stage A: assay qualification and dose range

Use third-instar larvae from a single defined genetic background. Match food, age, density, temperature, circadian time, handling, solvent, and exposure duration across arms. Randomize larvae and plate/well positions; blind image and movement scoring to treatment. Run independent biological cohorts on separate days.

First qualify the vehicle assay and establish an imidacloprid dose-response around the published 2.5 ppm, including vehicle and lower/higher concentrations. The exact range must be set from local formulation and pilot tolerability; do not assume that concentration in food equals internal dose. Choose a concentration that gives a robust, non-saturating assay phenotype without excessive mortality. Use the same exposure duration as the endpoint-specific published protocol where possible, and record deviations.

**Controls**

1. Vehicle only, time-matched.
2. Imidacloprid only at the selected qualified dose.
3. NACA only, at a literature-informed starting condition, after confirming vehicle compatibility and baseline tolerability.
4. Imidacloprid + NACA, with pretreatment/exposure schedule fixed before the screen.
5. Candidate only, at each tested concentration.
6. Imidacloprid + candidate, at each tested concentration.

Use the published NACA concentration/schedule only as a starting replication condition (the study reports 75 µg/mL pretreatment for 5 h for the larval rescue assays); verify against the paper’s methods and local protocol before execution. NACA tests downstream rescue capacity and does not establish candidate target engagement.

## 4. Stage B: blinded candidate screen

Screen a small, chemically diverse, provenance-documented pilot panel first. Require each candidate record to include structure/identifier, purity, source, solubility, solvent, known fly/human target evidence, dose rationale, and uncertainty. Exclude compounds that precipitate, alter the vehicle, or cause substantial candidate-alone toxicity at the tested exposure.

For every candidate, test a concentration series in candidate-only and imidacloprid + candidate arms. Keep controls on every experimental block. Randomize condition positions, blind acquisition/analysis, and repeat independent cohorts. Do not select the best concentration using the held-out confirmation cohort.

## 5. Endpoints

### Primary behavior endpoint

Measure larval movement using a validated Wiggle Index-style assay. For each independent experimental unit, calculate the movement ratio between the prespecified post-exposure time and the matched time-matched vehicle control. Preserve raw images/video and per-well measurements. If the same larvae are measured before and after exposure, treat that as a repeated measure and retain a time-matched vehicle arm to estimate handling/time effects.

### Primary neural endpoint

In a separate, genotype-matched cohort expressing a calcium indicator in the prespecified nAChR-expressing neuronal population, measure calcium response over a fixed baseline and exposure window. Normalize to the reference fluorophore if using a dual-reporter design, and use the fly/preparation as the biological unit; cells/ROIs are nested observations, not independent replicates. Prespecify response amplitude, duration/AUC, baseline drift, motion/artifact rejection, and acquisition settings.

### Safety and secondary endpoints

Record acute survival/recovery and developmental progression at prespecified times. Optional ROS or metabolic endpoints may be added only as secondary/mechanistic readouts with separate controls and analysis plans. Do not infer seizure, paralysis, or clinical benefit from movement or calcium alone.

## 6. Analysis and hit decision

Estimate candidate effects against both vehicle and imidacloprid-only controls, with experimental day/plate as blocking or random effects where appropriate. Report raw distributions, effect sizes, confidence intervals, exclusions, and multiplicity-adjusted tests. The biological unit is the independently treated fly or well, not each video frame, neuron, or ROI.

A candidate is a **fly-screen hit** only if it:

- changes the prespecified imidacloprid phenotype by at least the minimum meaningful effect set in the preregistration;
- shows a concentration-response pattern compatible with the hypothesis;
- does not produce the same apparent “rescue” by suppressing movement in candidate-only controls;
- does not create unacceptable survival/development effects under prespecified criteria; and
- reproduces in a blinded, independent confirmation cohort.

Call neural/behavior endpoints concordant only if the prespecified joint rule is met. If behavior improves but neural signal does not, or vice versa, report the split result and investigate rather than averaging endpoints into one score.

## 7. Mechanism and follow-up

For confirmed hits, test target engagement separately from phenotype:

- compare wild type with a justified target perturbation or resistant allele, plus genetic-background controls;
- test the proposed target in a receptor-specific functional assay when available;
- distinguish direct nAChR modulation from downstream antioxidant/metabolic rescue (NACA is the latter comparator in this study);
- assess candidate-alone effects and plausible off-target pathways;
- measure internal exposure/metabolism when a dose-response interpretation depends on it.

Only after replicated fly phenotypes and target evidence should a candidate be prioritized for another model system. Fly pharmacokinetics, metabolism, receptor composition, and tissue exposure differ from mammals; a fly hit is not a human drug recommendation.

## 8. Simulator integration and falsification rules

Before using this screen to train or score the connectome model, add explicit postsynaptic receptor identities, receptor-specific compound actions, dose/exposure assumptions, and a calcium observation model. Validate those components on hand-computable toy circuits and independent neural data. Do not encode a desired behavior outcome as transmitter gains.

For each experimental condition, freeze a model prediction before revealing the held-out result. Compare like with like: modeled calcium proxy to measured calcium only after calibration, and modeled motor output to behavior only after an independently established mapping. Keep simulation throughput separate from predictive accuracy. A failed direction is a model miss and must remain in the report.

## 9. What can be tested now

The repository contains Denecke et al. genotype-level movement data and an analysis script. Re-running that analysis tests parsing and the descriptive historical comparison only. It cannot test the neural endpoint, NACA rescue, candidate compounds, matched-vehicle effect, or a prospective held-out prediction. Those require new experimental data.

Offline check:

```bash
.venv/bin/python data/references/denecke_2017_imidacloprid_dgrp/analyze_wiggle_index.py
.venv/bin/python -m unittest discover -s tests -v
```

Expected dataset checks from the currently stored source: 37 held-out genotype-dose pairs at each of 25 and 100 ppm; median 60-minute/0-minute movement ratios approximately 0.716 and 0.565, respectively; all held-out ratios below 1. These are not vehicle-corrected causal estimates and are not a successful simulator validation.

### Recorded computational run

On 2026-09-30, the CLI ran a 100-step neutral model control and the stored ACh-only imidacloprid gain hypothesis through `gpu-wait`, using `screen_compound.py --steps 100 compounds/imidacloprid.json`. The receipt, exact stdout/stderr, engine/profile hashes, GPU pre-run state, and queue statistics are in `.work/neuropharmacology/compound-protocol-run-20260930/`. The run exited 0 in about 3.8 seconds. The output reported baseline active=13,978.6 and descending=250.8; the profile changed active count by +245.0%, descending activity by +250.4%, and synchrony by +85.4%, with reported step latencies 3.215 ms (control) and 3.443 ms (profile).

This is only a simulated gain perturbation, not any of the wet-lab treatment arms. The engine cannot represent imidacloprid concentration, receptor occupancy/desensitization, calcium, ROS, NACA, candidate-alone exposure, larval movement, or rescue. Therefore no NACA or compound-rescue comparison was run, and no biological outcome is inferred from this execution.

### ACh gain sensitivity sweep

A separate 100-step sweep ran neutral gain (1.0) and ACh gains 1.5, 2.0, 3.0, and 4.0, leaving every other gain neutral. Each condition was compared with a fresh neutral-control run. Full parsed telemetry and receipt are in `.work/neuropharmacology/gain-sweep-20260930/results.json`; raw engine stdout/stderr are in its `raw/` subdirectory. Modeled active-neuron counts were 13,979; 24,174; 33,951; 48,228; and 59,971, respectively. Descending activity was 251; 555; 762; 879; and 921. Thus the chosen gain settings produce different, increasing model outputs. These are **gain-sensitivity results**, not a chemical dose-response, receptor occupancy curve, toxicity result, or prediction of fly behavior.

## References

- Martelli et al. (2020), *Low doses of the neonicotinoid insecticide imidacloprid induce ROS triggering neurological and metabolic impairments in Drosophila*. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7568275/).
- Denecke et al. (2017), *Multiple P450s and Variation in Neuronal Genes Underpins the Response to the Insecticide Imidacloprid in a Population of Drosophila melanogaster*. [Primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5596007/); local raw-data provenance is documented in `data/references/denecke_2017_imidacloprid_dgrp/README.md`.
