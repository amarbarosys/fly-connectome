# Fly World agent evidence

2026-09-30 | P0 implementation started | Owner: wDV:p2

The first execution packet now has a receipt-backed evidence passport, CPU neural checkpoints, a full-connectome replay gate and a small independent scalar oracle. A later CPU embodied fixture also checkpoints body physics, controller/RNG, readout learning, organism/world state and engineered motivation. P0 remains partial for vision history, cross-instance embodied restore, GPU worker state and biological mapping/units.

## Fresh verification

[CPU gate receipt](../.work/agent/p0-evidence/replay-20260930-v2/metrics.json): PASS on 139,255 neurons and 10,572,269 edges, about 1.95 seconds wall time in the recorded run.

- Five exchanges after a nonzero checkpoint reproduced responses and all four mutable state arrays exactly.
- Independent CPU instances and a zero-input second lane stayed isolated.
- Changed effective graph weights were rejected before mutation.
- A corrupted finite state value was rejected by its checksum before mutation.
- A 16-tick independent four-node scalar oracle checked signed edge propagation, leak, threshold/reset, trace and recruitment, with voltage tolerance 1e-7. The graph is embedded in the existing reference dimensions; unused neurons stay inactive.

This initial neural gate is CPU software evidence and launched no viewer. Later embodied and actual-controls receipts are recorded below.

## Capability ledger

Machine-readable [evidence passport](../.work/agent/p0-evidence/evidence-passport.json) contains hashes, original receipt fields, available source-hash comparisons and missing evidence. Older receipts are recorded evidence and were not rerun by this task. Where old source hashes differ, the receipt does not establish current-code behavior.

| Capability | Mode | Supported evidence | Open claim |
|---|---|---|---|
| Neural interface | Engineering agent | Recorded CPU/GPU parity and lane isolation | Current GPU implementation equivalence and biological semantics |
| Body interface | Engineering agent | Recorded 0.6-second intact/null/disconnected comparison | Sustained stability and competence |
| Decoder learning | Engineering agent | Physical-ingestion-reward steering: 15/16 held-out successes after fresh-process restart, versus 10/16 shuffled and 0/16 straight/odor-absent | Architectural advantage, repeated closed-loop competence and autonomous learning |
| Vision | Engineering agent | Recorded rendered images drive coarse ports | Recognition, retinal fidelity and visual task competence |
| Wall response | Engineering agent | Recorded engineered contact-turn test | Learned avoidance and general recovery |
| Ten independent arenas | Engineering agent | Bounded three-seed collection 4.60-4.67x faster; equal collection-time held-out scores 29/30 vs 14/30 serial; read-only evaluator with 10.4% measured overhead | Repeated decisions, broader tasks, end-to-end timer-capped efficacy and whole-body GPU execution |
| Worker allocation | Engineering agent | Recorded bounded worker allocation reduction | Long-run leak freedom and scale |
| Learned skill retention | Engineering agent | Food success 16/16 after obstacle training with eight rehearsed examples; exact saved-model reload | Retention without rehearsal, delayed recall and general lifelong memory |
| Flight | Engineering agent | Recorded short pretrained airborne rollout | Neural control, takeoff and landing |
| Neural checkpoint/replay | Engineering agent | Fresh CPU exact-state gate and CPU no-vision embodied replay with learning updates | Vision/GPU and cross-instance episode replay |
| Teams, world evolution, gladiator skills | Engineering agent | Planning proposals | UNVERIFIED |
| Biological prediction | Biological model | No matched validating endpoint | UNVERIFIED |

## Reproduction

From the project root, use a new output directory for each run:

```sh
.venv/bin/python -m agent.verify_replay --out .work/agent/p0-evidence/replay-new
.venv/bin/python -m agent.evidence_passport --replay .work/agent/p0-evidence/replay-new --out .work/agent/p0-evidence/passport-new.json
```

The replay command refuses an existing directory and needs the existing CSR and sensory-port assets. It runs CPU-only. Artifacts include inputs, responses, effective graph/port identity, source hashes, numerical state and metrics. A failure exits nonzero and does not produce a PASS receipt.

`agent/neural_checkpoint.py` saves voltage, spikes, filtered spike traces and recruitment history. Effective CSR arrays, ports, descending IDs and dynamics source hash bind the checkpoint to the reference implementation. Restore validates every field, dtype, shape, state range and checksum before changing the destination. Publication is atomic. This is a local scientific checkpoint format, not a security boundary for hostile archives.

## Remaining P0 work

- Extend verified same-instance CPU embodied replay to cross-instance restore, vision history and resident GPU worker state.
- Finish mapping/units coverage and named sensory/motor gaps before biological interpretation.
- Then execute P1 sustained balance and live reset verification, with finite physics and recovery metrics.

[Master plan](plans/FLY_WORLD_MASTER_PLAN.md) and [100 experiments](plans/FLY_WORLD_100_IDEAS.md) remain the program map.


## CPU embodied replay and actual reset, 2026-09-30

[Embodied receipt](../.work/agent/p0-evidence/episode-full-gate-v4/metrics.json): three nonzero synthetic-teacher readout updates in a 20-exchange prefix; exact events and final body/controller/neural/readout/world state across a 30-exchange replay. Incompatible archives were rejected. This is a no-vision persistence fixture, not a learning-performance claim. The teacher explicitly stimulates all eight ports and reinforces FEED; the old IDLE-action fixture did not exercise plasticity.

[Actual UI receipt](../.work/agent/p1-body-gate/ui-reset-v2/metrics.json): physical XWayland mouse clicks exercised the real pause/reset/resume/stop buttons with CPU neural dynamics and software rendering. Full body/controller snapshot restored, nonzero neural arrays cleared, learned readout weights preserved, and ten post-reset exchanges executed. UI reset is body/brain reset; world/organism and learned decoder are retained.

## Bounded learned steering and retention: 2026-10-01

[Protocol and receipts](plans/ODOR_MEMORY_PILOT.md): odor steering PASS; physical obstacle-avoidance improvement FAIL against the strong straight baseline; food retention with explicit rehearsal PASS. Independent audits validate raw trajectories and reward/intake accounting. GPU neural replay PASS on 24 recorded input cases, not full GPU physics or live UI. Models remain separate experimental artifacts; P2 is partial.

## Ten-fly capacity, 2026-10-01

[Measured report](plans/TEN_FLY_CAPACITY.md): bounded ten-arena feedback trial and independent 2,162-exchange audit PASS. Six CPU body workers plus GPU neural batching achieved 0.873 aggregate fly-seconds/wall-second. Vision, real-time ten-fly viewing and improved learning efficiency remain unverified.

## Spectator and matched learning budgets, 2026-10-01

[Report](plans/SPECTATOR_PARALLEL_LEARNING.md): one visible read-only evaluator, atomic snapshot reload, and shutdown/disconnect regressions PASS. Three training seeds, ten unique shared held-out layouts: equal-experience policies both 29/30; ten-arena collection 4.60-4.67x faster. Equal collection-time completed-sample prefixes scored 14/30 serial and 29/30 ten, using frozen offline choices over actual physical turn rollouts. Independent 16,080-exchange training audit PASS. Weak shuffled labels in the third control, cached spectator sensory reference, and the one-decision/prescribed-feed scope limit interpretation.

## SkillOpt patch bridge and extra asset activity, 2026-10-01

[Report](plans/SKILLOPT_FLY_PILOT.md): assistant-proposed archived SkillOpt patch engine pilot, 150.12-second video/neural recording, five matched selection pairs (candidate 5/5, baseline 3/5). CPU-shadow neuron-ID audit and GPU feature/count parity PASS. Exact-ID sampled skeleton/synapse activity joins added; atlas/physiology identity crosswalks remain open. Autonomous optimizer loop and final generalization are unverified.

## Calcium measurement pilot, 2026-10-01

[Protocol and results](plans/CALCIUM_VALIDATION.md): the initial pilot failed its prediction gate. The source-informed follow-up preserved 288 source-clock-aligned records and accumulated 12,304 neural updates, with 47 annotated EPG neurons. Final median R² 0.002888 versus heading baseline -0.46105, time-shift p=0.70: prediction gate FAIL. No accepted observer or active neural dynamics update. Optical/biological clock calibration and ROI-to-neuron identity remain unverified.

## Control and scientific validity audit, 2026-10-01

[Control audit](plans/CONTROL_AUDIT.md): explicit natural/diagnostic profiles, local MCP with acknowledged pause/resume/reset/step/session stop, managed queued launch/cancellation, unique run directories and final-frame consumption guards. CPU and GPU-brain/CPU-body control integration PASS; GPU body physics and GUI interaction are not established by these tests. MCP registered in Codex; current native session catalog not reloaded.

[Sensory/state gates](plans/SENSORY_VALIDITY_GATES.md): CPU mechanics PASS exposes scientific input/state validity failures. A saved warm state fires with zero sensory input; disconnected controls do not. Distinct static retinal cue bearings collapse to identical coarse visual ports. No parameters or real final calcium targets were changed/refitted. Training expansion waits for a defensible spatial/transduction/state calibration.
