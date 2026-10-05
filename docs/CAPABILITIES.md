# Current capabilities

Status describes the cited implementation and gate only. It does not imply biological validation.

## BODY

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| CPU embodied floor and wall gate | PASS | 20/20 floor trials upright; 19/20 wall escapes | `docs/plans/GPU_BODY_PORT.md`; `.work/agent/p1-body-gate/full-verification-cpu-ui.json` | `./.venv/bin/python -m agent.verify_body_gate` (supply receipt paths) |
| RX 7900 XTX GPU body physics | UNVERIFIED | 1 diagnostic step finite; diagnostic qpos NaN by 1,000 steps | `docs/plans/GPU_BODY_PORT.md`; `.work/agent/gfx1100-probe/result.json` | Probe in progress; no validated runtime entrypoint |
| Own GPU body engine (MuJoCo 3.9.0 port, any MJCF) | PARTIAL (S1, S3, floor + cube contacts PASS; noslip, non-plane contacts missing) | actuated fly matches MuJoCo within 1e-5 or its float32-input floor on every field, mutants caught; all 55 floor pairs (feet-only sank in MuJoCo too, `00a58e6`); K steps per launch, K=10 byte-equal to 10 x K=1 (`c5c478c`); floating origin per animal (`a03e700`); closed 40 mm cube of 6 planes, states on every face (`5acde4e`); endless terrain as a function (`world.mojo`, slope + hashed gradient noise, 32 B, `1ba87b5`) contacted as a tangent plane: fly PASS on a ramp vs MuJoCo's plane twin, noise lookup <0.3% of step (`f57f70c`, `check.sh terrain`; not mjc_ConvexHField, wrong where terrain bends inside one mesh); fly stands on 0.1 mm bumps, collapses on 0.3 mm (`33ec8f7`, `live.py upright`); fly + spider in one model (`f415c91`, `check.sh world` PASS); 64 animals 23.6 ms per step on the floor. Standalone plume content: seeded food/decoy descriptor and antenna sampler; maps.py oracle max abs error 1.8e-7, GPU batch 4e-8 with seed-0 scenario positions, reversal/depletion/seed checks and mutants PASS; live/render hook pending owner integration (`644c054`, `19664b3`) | `agent/body_engine/README.md`, `docs/plans/BODY_ENGINE.md`, `docs/plans/WORLD.md`; commits `ccd9695`..`a03e700`, `644c054`, `19664b3` | `gpu-wait run --priority 90 --timeout 3600 -- agent/body_engine/check.sh STAGE` (kinematics crb factorM rne step contact actuated loop cube; `BODY=path.xml` for another body); `.venv/bin/python -m agent.body_engine.world_content_check` |
| Live view (engine + renderer, one process) | PASS | stand broke at noslip port 59abe8d (motion 0.127, MuJoCo trembles too), fixed 74485d5: engine noslip 0, standing pose from the noslip-5 model, motion 0.0105; upright (`agent/body_engine/stand.sh`): ramp 0.1/0.05 thorax 0.485 mm (was 0.001: start lifted 0.57 mm over the highest ground within 4 mm, MuJoCo slid and lay down too), 0.1 mm bumps 0.486, flat 0.484, flipped FAIL on both; first frame byte-equal to render.mojo; stands 300 frames vs a feet-only control that falls apart; 10,000 frames no leak (VRAM 151.8 MB flat); cube walls drawn; terrain drawn by ray march, amplitude-0 equals the flat floor within 1 LSB (`9fbe141`, `terrain_render_check`) | commit `f086e5b`, `5acde4e` | `.venv/bin/python -m agent.body_engine.live check`; `... live watch 10 [cube] [spider]` (under gpu-wait) |
| Jumping-spider body (generated) | PARTIAL | `models/spider.py`: 23.5 mg, 64 dofs, 58 servos, stands in MuJoCo; engine PASS kinematics..step and cube; contact/actuated FAIL on qfrc_constraint only (1.5e-05 to 5.2e-05 vs 1e-05; qacc passes); anatomy approximate | commits `615d2af`, `476d6c1` | `.venv/bin/python -m agent.body_engine.models.spider`; `BODY=.work/agent/body_engine/spider-body/spider.xml check.sh STAGE` |
| Own renderer on the engine's poses | PASS (steps 1-5) | silhouette IoU vs mujoco.Renderer 0.995-0.998 in 3 views; 1 fly 960 x 540 0.57-1.0 ms per frame; 10,000 frames without leak (fds 7/7, VRAM flat 152 MB) | `docs/plans/RENDERER.md`; commits `0ad21a9`, `905d2bc`, `86fb4d2` | `env MUJOCO_GL=egl python -m agent.body_engine.render_check`; `python -m agent.body_engine.verify_render_leak` |
| Game state and resource-event accounting | PARTIAL (standalone module; engine hook pending) | 16 B/animal; standalone work drain, tagged hazard once, rest, absorbing death, duplicate-event and shared-resource replay pass; 4 B/resource in CPU oracle; no GPU event scratch/layout or timing claimed | `docs/plans/GAME_LAYER.md`; standalone gate `python -m agent.body_engine.game_state_check` | `python -m agent.body_engine.game_state_check` |
| Spectator camera and HUD layouts | PARTIAL (standalone units 1-2) | merged two-root sphere framing passes landscape and portrait; deliberate off-center target mutant rejected; energy/health/stamina fills 90/30/60 px and source descriptor marker fixture pass; no live hook or moving-pose clip yet | `docs/plans/RENDERER.md`; commits `77aee00`, `878eeb4` | `.venv/bin/python agent/body_engine/spectator_check.py` |

## SENSES

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Optic-column to arbor geometry correspondence | PASS | 4,530 neurons; r=0.914 left, 0.905 right; shuffled means 0.007 and 0.003 | `.work/agent/column-geometry/verification.json`; `docs/plans/SPATIAL_VISION.md` | `./.venv/bin/python -m agent.verify_column_geometry` |
| Retinal angular calibration | UNVERIFIED | 751/778 right-eye rays in view; 27 outside | `docs/plans/RETINAL_CALIBRATION.md` | `./.venv/bin/python -m agent.verify_retinal_geometry` (reference input required) |
| Dim-stripe visual response | broken | Maximum 0.139, below spike threshold 1 | `docs/plans/SENSORY_VALIDITY_GATES.md` | `./.venv/bin/python -m agent.verify_sensory_recurrence --output <receipt>` |

## BRAIN

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Brain settings (one setting for every intervention) | PASS | stimulate / inject / silence / cut / gain / modulation by FlyWire selectors on two device primitives; silence == hand-removed edges (0 cells differ, mutant 191), inject == one-cell model, GPU = CPU exact, live change == fresh run; 0 B per cell or edge, sniff time unchanged | commit `dca237d`; `docs/plans/BRAIN_CONTROLS.md` | `.venv/bin/python -m agent.verify_settings cpu`; `gpu-wait ... -m agent.verify_settings gpu` |
| Full-brain recording | PASS | every neuron's count every window (138,639 neurons), 0 mismatches vs the worker's own readout | commit `e6d18db` | `gpu-wait ... -m agent.verify_record` |
| Slow neuromodulation (DA / OA / 5-HT levels: gain + current) | PASS (mechanism); sugar -> PAM still 0 | GPU = CPU exact on/off; octopamine level outlasts its drive (1.13x -> 0.99x by 1.5 s), mutant caught; only 53 OA -> PAM synapses, so synapse-routed modulation cannot carry the sugar teacher | commits `b10bd14`, `8f20a1f`, `f4fafd7` | `gpu-wait ... -m agent.verify_modulation` |
| Odor steering with fixed real connectome and trained readout | FAIL (gate) | 56/64 trained, but only 20.3 points over 43/64 shuffled reward against the frozen 25-point gate; odor-absent and straight controls 0/16 on the sanity subset | `.work/agent/matched-baseline/scale-comparison-corrected.json`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.baseline_scale` (experiment runner; existing results are the cited evidence) |
| Specific wiring advantage over degree-preserving shuffles, on odor steering | FAIL | Real trails 4/5 shuffles; seed-cluster pooled difference -1.875 percentage points, p=0.7551 | `.work/agent/paired-significance/significance-corrected-seedcluster.json`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.verify_parallel_learning_trial` is a separate learning verifier |
| Sparse odour code in Kenyon cells (body-free bench) | PASS at gain 35 | Normalised gain 35: KC active 6.5% / 7.6% / 1.9% for odours A/B/C, pairwise Jaccard at most 0.027; raw weights flood to 79% with Jaccard 1.0 | `.work/briefs/science-path/REPORT-P1b.md`; `REPORT-P2.md` | `./.venv/bin/python -m agent.brain_bench` |
| Mushroom-body output and motor drive at sparse gain | FAIL | 0/96 MBONs fire at gains 30 and 35; 0/1,303 descending neurons fire at gains 25 to 40 | `.work/briefs/science-path/REPORT-P2.md` | `./.venv/bin/python -m agent.verify_sparse_learning` |
| Sparse KC code with live MBON output together | FAIL | No row-normalisation exponent (p 0.5 to 1, 30 settings) gives both: KC is either sparse with 0 MBONs or about 70% active. With a 2% per-tick KC cap at gain 50 to 60, MBONs fire odour-specifically (Jaccard at most 0.44) and DNs fire, but cumulative KC activity is 12 to 25% against a 10% limit | `.work/agent/brain-bench/sweep-p1c.json`; `sweep-p1c-cap.json` | `./.venv/bin/python -m agent.brain_bench --p1c` / `--p1c-cap` |
| Per-class gain operating point (APL, MBON, DN) | FAIL (nearest 17/18) | Operating-point model: APL needs 6.7x input, MBON 10.8x, DN 83x at gain 35. 162-point class-gain grid, 6 odours x 3 jitter seeds: every criterion reachable, none jointly; MBON x5 / DN x80 passes all but one weak odour at 0.8% KC. APL never fires there | `.work/agent/brain-bench/sweep-p1d.json` | `./.venv/bin/python -m agent.brain_bench --p1d` |
| Shiu 2024 LIF model (CPU oracle + GPU) | PASS | Sugar GRNs 1 s: 16,984 spikes / 396 active vs Brian2 16,760 / 390; GPU per-cell counts identical to the oracle; 1 s x 5 trials in 2.5 s | `.work/agent/shiu/verification.json`; `verification-gpu.json` | `./.venv/bin/python -m agent.verify_shiu [--gpu]` |
| Real wiring vs matched controls, Shiu odour coding | PASS | Odour decode at 30 ms: real 0.94 vs block shuffles 0.22-0.56 (chance 1/6; arms without self-loops or repeated pairs, `repair_edges` unique), sign-flip p=2.3e-05; degree-preserving shuffles never activate KCs | `.work/agent/shiu/g1.json`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.shiu_g1` |
| Same test, second animal (male CNS v1.0) | PASS | 165,733 neurons, 6.24M edges (>= 5 synapses). Real vs 5 block shuffles, 18/18 columns at 30 ms, p=7.6e-06; decode at 20 ms real 0.94 vs block 0.28-0.61; flood decode real 1.00 vs block 0.28-0.56. Degree shuffles again silence KCs (<= 0.27%), so their higher late scores are an activity artefact | `.work/agent/shiu/g1-mcns.json`; `agent/mcns.py` | `./.venv/bin/python -m agent.shiu_g1 mcns` (data: flat-connectome files in `.work/mcns/`) |
| Shiu DN output carries odour identity, not side or concentration | PASS (probe) | DN 30-50 ms decode real 1.00 vs block 0.56 (2 odours x 4 concentrations); no lateralised DN (max z 3.0 over 1,291); DN output +15% over 40x ORN rate | `.work/agent/shiu/dn_*.py` (probe scripts) | GPU, ad hoc |
| Closed-loop odour choice, real vs controls (FlyWire) | NOT SIGNIFICANT | Margin real +0.239 vs block +0.084 (29/48, p=0.19), shuffled +0.059 (30/48, p=0.11); random-walk variance about +-0.15 swamps the effect at 48 columns. All 11 arms in one resident worker, 6.0 min | `.work/agent/shiu/loop-flywire.json`; `agent/shiu_loop.py` | `./.venv/bin/python -m agent.shiu_loop` |
| Closed loop on four maps, real vs controls (FlyWire) | NO WIRING BENEFIT | Score real vs block mean (p): tmaze +2.90 vs +2.52 (0.80), open +1.50 vs +1.05 (0.22), plume +0.85 vs +0.34 (0.25), maze -2.25 vs -2.55 (0.61); vs shuffled p 0.11 to 0.83. Maze degenerate: 0 food in every arm (decoy beside start captures walkers). 48 columns, about 5 min per map | `.work/agent/shiu/loop-flywire-{tmaze,open,plume,maze}.json`; `agent/maps.py`, `tests/test_maps.py` | `./.venv/bin/python -m agent.shiu_loop flywire <map>` |
| Visual side in DNs, real vs controls (FlyWire) | REAL ONLY (structure) | One-side looming (LC4+LPLC2): 30-50 ms real 17 DNs \|z\|>4, max \|z\| 13.0 in mirrored DNp04/02/05/11; block 0-2 DNs, max \|z\| 3.2-4.5; side decode saturates (1.00) in both. One-eye R1-6: 0 DN spikes in all arms | `.work/agent/shiu/visual-side.json`; `agent/visual_side.py` | `./.venv/bin/python -m agent.visual_side` |
| Shiu KC flood source and fix (FlyWire) | FIXED (antennal lobe) | ALPNs 66% active, A/B Jaccard 0.96 at 30 ms. KC->KC and DAN->KC off: KC 65% to 51%; APL->KC x4: KC 6% but J 0.81; ALLN inhibition x12 at 200 Hz: KC 6.2%, J 0.19, MBONs 45, DNs 190 (x10: 9.1%, J 0.23) | `.work/agent/shiu/apl-sweep.json`; `agent/shiu_apl.py` | `./.venv/bin/python -m agent.shiu_apl` |
| Mushroom-body reward specificity on Shiu (FlyWire, ALLN x12) | PARTIAL (1/3 pairs) | Reward A depresses KC->MBON from A's KCs; MBON spikes lost A vs B vs sham: DL5+DC1+VA7l pair 26.0 / 0.1 / 0.6 PASS; other pairs FAIL on weak A drive (6, 16 MBON spikes). Authors' graph: FAIL at every window (flood at readout) | `.work/agent/shiu/learning-early.json`; `agent/shiu_learning.py` | `./.venv/bin/python -m agent.shiu_learning` |
| Male CNS eye map (viewing direction per visual neuron) | WORKS | 3,251/4,108 photoreceptors placed by flyverse's column method (MIT); eyes about 125 x 145 deg; LC4 / LPLC2 fields from inputs alone land on their soma side's eye for 100% / 97% of cells; LC9 1/219 (inputs carry no column) | `.work/agent/shiu/eye-map-mcns.npz`; `agent/eye_map.py` | `./.venv/bin/python -m agent.eye_map` |
| FlyGym 2.x body (CPU MuJoCo 3.9, imported) | WORKS (CPU) | CPG fly walks 13.10 mm per simulated second at 0.41x real time. GPU path blocked on versions: FlyGym pins warp-lang 1.14 / mujoco_warp 3.9; ROCm forks are 1.13 / 3.8.1, gfx942 only | `tools/flygym_smoke.py`; venv `.work/flygym-venv` | `.work/flygym-venv/bin/python tools/flygym_smoke.py` |
| Plastic KC factors + reward (learning on the device) | WORKS | CPU oracle and resident GPU agree exactly over sniff, reward, sniff; reward on VM2+DM1+VA2 drops its avoid-MBON spikes 176.8 -> 7.1 and lifts valence 0.24 -> 0.96, partner 0.84 -> 0.88 | `.work/agent/shiu/verify-plastic.json`; `agent/verify_plastic.py` | `./.venv/bin/python -m agent.verify_plastic` |
| Punishment plane (KC -> approach MBONs) | WORKS | Punished A: approach-MBON 38.4 vs 288.1 twins, valence +0.24 -> -0.56, B 0.84 -> 0.89; CPU = GPU exact, reward factors untouched | `agent/verify_pain.py`; commit `b49c92f` | `./.venv/bin/python -m agent.verify_pain` |
| Sugar reaches PAM under hunger gain | FAIL | 0 PAM spikes at hunger 0, 0.5, 1 with sugar output and PAM input x3; octopamine VPM3/4 give PAM 29 of 109,448 excitatory synapses | `.work/agent/shiu/presets/hunger.json`; commit `06232c3` | `./.venv/bin/python -m agent.presets hunger` |
| Degree structure versus in-degree-only random graphs | PASS | Real advantage +18.75 percentage points, seed-cluster pooled p=0.001472 | `.work/agent/paired-significance/significance-corrected-seedcluster.json`; `docs/plans/CONNECTOME_BENEFIT.md` | See cited comparison receipt; no standalone verifier |

Correction 2026-10-02: p-values recomputed with seed clusters (mirrored layouts are not independent); old pooled values 0.6990 (shuffles) and 0.000489 (random) are now 0.7551 and 0.001472; conclusions unchanged: specific wiring does not beat degree-preserving shuffles, while the real graph remains ahead of the random-column family.

## MEMORY

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Learned one-decision odor steering readout | PASS | 15/16 in the pilot; current scaled real arm 56/64 | `docs/plans/ODOR_MEMORY_PILOT.md`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.verify_odor_memory_receipt --output <receipt>` |
| Continuous state across calcium replay windows | broken | Brain state resets at each measurement window | `docs/plans/CONTROL_AUDIT.md`; `docs/plans/CALCIUM_VALIDATION.md` | `./.venv/bin/python -m agent.verify_calcium_validation` checks the replay validation receipt |

## LEARNING

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Reward-trained decoder/readout with fixed connectome weights | PASS | Learning changes readout; connectome weights stay fixed | `docs/plans/ODOR_MEMORY_PILOT.md`; `docs/plans/CONNECTOME_BENEFIT.md` | `./.venv/bin/python -m agent.odor_memory_gate` |
| Dopamine-gated KC to MBON plasticity (isolated rule) | PASS | Hand-set traces, one LTD event: exactly the eligible edges move, all down, for both the KC x MBON and KC-only rules | `.work/agent/mb-plasticity/verification.json` | `./.venv/bin/python -m agent.verify_mb_plasticity` |
| Dopamine-gated KC to MBON plasticity (through the odour chain) | FAIL | At the P1d point (gain 35, MBON x5, DN x80), KC-only eligibility, reward as depression: A-active KC edges change 23.9x more than B-active; MBONs fire 15/19/4; but MBON response after reward moves more for B (0.187) than A (0.034) | `.work/agent/sparse-learning/result.json` | `./.venv/bin/python -m agent.verify_sparse_learning` |
| Food learning in the closed loop (reward on bites) | PASS | Late score learners -1.76 vs twins -6.90, 28/48 pairs higher, exact sign-flip p=0.0049; food bites 378 vs 175 by trial 6 | `.work/agent/shiu/learning-loop-tmaze.json`; commit `5eda2db` | `gpu-wait run --timeout 3600 -- ./.venv/bin/python -m agent.learning_loop` |
| Biological learning in the connectome | UNVERIFIED | Not measured | `docs/plans/CONNECTOME_BENEFIT.md`; `docs/plans/FLY_WORLD_MASTER_PLAN.md` | No biological-learning implementation |
| Pain learning in the closed loop (trap avoidance) | PASS | Credit to the smell at contact: late trap contacts 2.17 vs twins 11.92, 31/32 pairs, p=9.3e-10 (relief arm 1.72, 29/32, p=6.5e-09) | `docs/plans/LEARNING_LOOP.md`; `.work/agent/shiu/pain-loop-tmaze.json` | `gpu-wait run --timeout 3600 -- ./.venv/bin/python -m agent.pain_loop` |
| Parallel arena experience collection | PASS | 4.60 to 4.67x faster collection; equal experience 29/30 for both policies | `docs/plans/SPECTATOR_PARALLEL_LEARNING.md`; `.work/agent/spectator-learning/learning/comparison.json` | `./.venv/bin/python -m agent.verify_parallel_learning_trial` |

## TASKS

| Capability | Status | Number | Receipt | Entrypoint |
|---|---|---:|---|---|
| Odor-guided food task | PASS | 15/16 pilot layouts; odor-absent control 0/16 | `docs/plans/ODOR_MEMORY_PILOT.md`; `.work/agent/matched-baseline/scale-comparison-corrected.json` | `./.venv/bin/python -m agent.odor_memory_gate` |
| Repaired obstacle avoidance | PASS | With 40 rehearsed food examples: 14/16 vs 6/16 straight; food retained 16/16 before and after avoidance training (earlier run 12/16, food 14/16) | `.work/agent/odor-memory/sequential-p0-run/metrics.json`; `docs/plans/ODOR_MEMORY_PILOT.md` | `./.venv/bin/python -m agent.verify_odor_memory_receipt --output <receipt>` |
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
| GPU neural worker tick | PASS | 3.1 to 4.4x faster per tick, bit-identical to the previous kernel; CPU/GPU parity error 0.0 | commits `d065ff9`, `1eebbd2` | `.work/fly-research/arena-venv/bin/python -m agent.verify_neural` |
| Shiu resident sniff (96 columns, 1,000 steps) | PASS | 675 -> 453 ms, exact parity kept: rest skip, int16/uint8 state, deliver without refractory reads | commits `84af113`, `22932f8`, `95d2e0c`; `agent/profile_sniff.py` | `gpu-wait run ... rocprofv3 --kernel-trace --stats ... -m agent.profile_sniff` |
| GPU worker at 64 flies and alternate graphs | PASS | 64-fly build and 2-fly block-s7 graph (FLY_DATA_DIR) within 9.5e-7 of CPUBrain; spike words uint16 up to 16 flies, uint64 above | commit `704becb` | `.work/fly-research/arena-venv/bin/python -m agent.verify_worker_wide` |
| Body-free brain bench on GPU (WorkerBrain) | PASS | Odour drive, normalised and class gains, plasticity edits run on the GPU worker; ever sets identical to CPUBrain; P1d point 2.16 s CPU to 0.19 s GPU, 162-point grid 34 s; 161/162 rows identical to the CPU receipt (one flooded row differs 0.24%, same verdict) | `.work/neural-flies/verify-worker-brain.json`; `.work/agent/brain-bench/sweep-p1d-gpu.json` | `.work/fly-research/arena-venv/bin/python -m agent.verify_worker_brain` |
| Ten-fly full-loop capacity | PASS | Optimized aggregate 0.861 fly-seconds/wall-second; per-fly full loop 0.0873x real time | `docs/plans/TEN_FLY_CAPACITY.md`; `.work/agent/ten-fly/independent-verification.json` | `./.venv/bin/python -m agent.verify_batch_capacity` |

## Not yet / unlockable

- Compare wiring on harder tasks where it could matter: odor memory, vision, and looming.
- Lesion mapping is scoped to dependence under tested conditions; anatomical necessity remains unverified.
- GPU physics track closed; CPU MuJoCo stays the backend (`docs/plans/GPU_BODY_PORT.md`). Own engine (MuJoCo port) in progress: `docs/plans/BODY_ENGINE.md`.
- Engage MBON output at sparse gain (KC-trace eligibility rule, per-class gain), then the reversal task.
- Fix the dim-light vision response so the stripe can trigger spikes.
- Preserve continuous neural state across calcium replay windows.

## Does not claim

- No biological learning in the connectome is established; the learned component is a readout with fixed connectome weights.
- No clinical relevance is established.
- Passing engineering, parity, or task gates does not establish biological validity.
