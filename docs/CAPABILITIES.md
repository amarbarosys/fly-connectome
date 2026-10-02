# Current capabilities

Status describes the cited implementation and gate only. It does not imply biological validation.

## BODY

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| CPU embodied floor and wall gate | PASS | 20/20 floor trials upright; 19/20 wall escapes | `docs/plans/GPU_BODY_PORT.md`; `.work/agent/p1-body-gate/full-verification-cpu-ui.json` | `./.venv/bin/python -m agent.verify_body_gate` (supply receipt paths) |
| RX 7900 XTX GPU body physics | UNVERIFIED | 1 diagnostic step finite; diagnostic qpos NaN by 1,000 steps | `docs/plans/GPU_BODY_PORT.md`; `.work/agent/gfx1100-probe/result.json` | Probe in progress; no validated runtime entrypoint |

## SENSES

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Optic-column to arbor geometry correspondence | PASS | 4,530 neurons; r=0.914 left, 0.905 right; shuffled means 0.007 and 0.003 | `.work/agent/column-geometry/verification.json`; `docs/plans/SPATIAL_VISION.md` | `./.venv/bin/python -m agent.verify_column_geometry` |
| Retinal angular calibration | UNVERIFIED | 751/778 right-eye rays in view; 27 outside | `docs/plans/RETINAL_CALIBRATION.md` | `./.venv/bin/python -m agent.verify_retinal_geometry` (reference input required) |
| Dim-stripe visual response | broken | Maximum 0.139, below spike threshold 1 | `docs/plans/SENSORY_VALIDITY_GATES.md` | `./.venv/bin/python -m agent.verify_sensory_recurrence --output <receipt>` |

## BRAIN

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Odor steering with fixed real connectome and trained readout | FAIL (gate) | 56/64 trained, but only 20.3 points over 43/64 shuffled reward against the frozen 25-point gate; odor-absent and straight controls 0/16 on the sanity subset | `.work/agent/matched-baseline/scale-comparison-corrected.json`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.baseline_scale` (experiment runner; existing results are the cited evidence) |
| Specific wiring advantage over degree-preserving shuffles, on odor steering | FAIL | Real trails 4/5 shuffles; pooled difference -1.875 points, p=0.699 | `.work/agent/paired-significance/significance-scale-corrected.json`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.verify_parallel_learning_trial` is a separate learning verifier |
| Degree structure versus in-degree-only random graphs | PASS | Real advantage +18.75 points, pooled p=0.000489 | `.work/agent/paired-significance/significance-scale-corrected.json`; `docs/plans/CONNECTOME_BENEFIT.md` | See cited comparison receipt; no standalone verifier |

## MEMORY

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Learned one-decision odor steering readout | PASS | 15/16 in the pilot; current scaled real arm 56/64 | `docs/plans/ODOR_MEMORY_PILOT.md`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.verify_odor_memory_receipt --output <receipt>` |
| Continuous state across calcium replay windows | broken | Brain state resets at each measurement window | `docs/plans/CONTROL_AUDIT.md`; `docs/plans/CALCIUM_VALIDATION.md` | `./.venv/bin/python -m agent.verify_calcium_validation` checks the replay validation receipt |

## LEARNING

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Reward-trained decoder/readout with fixed connectome weights | PASS | Learning changes readout; connectome weights stay fixed | `docs/plans/ODOR_MEMORY_PILOT.md`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.odor_memory_gate` |
| Biological learning in the connectome | UNVERIFIED | Not measured | `docs/plans/CONNECTOME_BENEFIT.md`; `docs/plans/FLY_WORLD_MASTER_PLAN.md` | No biological-learning implementation |
| Parallel arena experience collection | PASS | 4.60 to 4.67x faster collection; equal experience 29/30 for both policies | `docs/plans/SPECTATOR_PARALLEL_LEARNING.md`; `.work/agent/spectator-learning/learning/comparison.json` | `./.venv/bin/python -m agent.verify_parallel_learning_trial` |

## TASKS

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Odor-guided food task | PASS | 15/16 pilot layouts; odor-absent control 0/16 | `docs/plans/ODOR_MEMORY_PILOT.md`; `.work/agent/matched-baseline/scale-comparison-corrected.json` | `./.venv/bin/python -m agent.odor_memory_gate` |
| Repaired obstacle avoidance | PASS | 12/16 vs 6/16 straight and 5/16 shuffled-reward; trained minimum upright 0.9904; food retention 14/16 vs 15/16 before | `.work/agent/odor-memory/calibration-v2/metrics.json`; `docs/plans/ODOR_MEMORY_PILOT.md` | `./.venv/bin/python -m agent.verify_odor_memory_receipt --output <receipt>` |
| Odor memory, vision, and looming as harder connectome-comparison tasks | UNVERIFIED | No comparative outcome claimed | `docs/plans/CONNECTOME_BENEFIT.md`; `docs/plans/P6_LOOMING_PROTOCOL.md` | No completed comparison entrypoint |

## SCIENCE TOOLING

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| L1 synapse linkage to FlyWire roster | PASS | 130,054,535 source rows; 139,255 roster neurons | `.work/agent/synapse-link/link.json`; `.work/agent/synapse-link/independent-verification.json`; `docs/plans/FIVE_LAYER_ASSETS.md` | `./.venv/bin/python -m agent.verify_synapse_link` |
| L2 skeleton linkage to FlyWire roster | PASS | 268,281,651 source rows; 139,255/139,255 roster neurons linked | `.work/agent/skeleton-link/link.json`; `.work/agent/skeleton-link/independent-verification.json` | `./.venv/bin/python -m agent.verify_skeleton_link` |
| Lesion and sweep condition-set generation | PASS | 24 lesion conditions; 33 sweep conditions | commit `e3afe06`; `fly-screen-rs/README.md` | `FLY_CONDITION_SET=lesion|sweep fly-screen-rs/target/release/fly-screen` |
| Lesion inference about anatomical necessity | UNVERIFIED | Condition sets are defined; no necessity claim | `docs/plans/SCREENING_REBUILD_PLAN.md` | No validated inference entrypoint |

## COMPUTE

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Rust/Mojo screening parity and throughput | PASS | 48 manifest rows match Python reference; 24.44x in paired benchmark | `docs/plans/SCREENING_REBUILD_PLAN.md`; `.work/runs/155567-1790771553562845391/receipt.json`; `.work/benchmarks/rebuild_20260930/paired_benchmark_receipt.json` | `fly-screen-rs/target/release/fly-screen` |
| Screening biological predictive validity | UNVERIFIED | Not established by parity or speed | `docs/plans/SCREENING_REBUILD_PLAN.md` | No biological-validation entrypoint |
| Ten-fly full-loop capacity | PASS | Optimized aggregate 0.861 fly-seconds/wall-second; per-fly full loop 0.0873x real time | `docs/plans/TEN_FLY_CAPACITY.md`; `.work/agent/ten-fly/independent-verification.json` | `./.venv/bin/python -m agent.verify_batch_capacity` |

## Not yet / unlockable

- Food recovery may improve with a larger rehearsal buffer.
- Compare wiring on harder tasks where it could matter: odor memory, vision, and looming.
- Lesion mapping is scoped to dependence under tested conditions; anatomical necessity remains unverified.
- GPU physics awaits a stable probe on the target device and faithful model support.
- Fix the dim-light vision response so the stripe can trigger spikes.
- Preserve continuous neural state across calcium replay windows.

## Does not claim

- No biological learning in the connectome is established; the learned component is a readout with fixed connectome weights.
- No clinical relevance is established.
- Passing engineering, parity, or task gates does not establish biological validity.
