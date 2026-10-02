# `fly-screen`

Rust owns catalog access, Hill gains, typed run manifests, process orchestration,
hashes, Parquet results, and receipts. `fly_batch_gpu.mojo` owns the resident
GPU simulation. Python remains an independent scientific oracle.

## Build and preflight

```sh
cd /home/<user>/Projects/mojo/fly-spmv
mojo build src/fly_batch_gpu.mojo -o .work/fly_batch_gpu

cd /home/<user>/Projects/science/fly-connectome/fly-screen-rs
cargo test --bin fly-screen
cargo build --release --bin fly-screen
FLY_DRY_ONLY=1 FLY_CONDITIONS=50 FLY_STEPS=100 target/release/fly-screen
```

The CPU preflight validates the worker's reported manifest, graph dimensions,
and nonempty unique descending-neuron count without creating a GPU context.
For a device-side kernel compile check without running a batch, use
`FLY_GPU_COMPILE_ONLY=1 target/release/fly-screen`; it queues a short 1 GB job.
Normal runs queue one batch job with a 22 GB reservation and 180 second timeout.

## Run

```sh
target/release/fly-screen
```

## Condition sets

`FLY_CONDITION_SET` picks what the conditions are:

| Value | Conditions | Needs |
| --- | --- | --- |
| `compound` (default) | dose ladders from a compound registry | `compounds/registry.parquet`, which now lives in the separate `fly-pharma` repository, so this set fails here by design |
| `lesion` | 3 controls, 6 single-slot silencings, 15 slot pairs, 24 total | nothing beyond the connectome |
| `sweep` | 3 controls plus each of 6 gain slots at 0.00x, 0.25x, 0.50x, 2.00x, 4.00x, 33 total | nothing beyond the connectome |

`lesion` and `sweep` perturb the same eight-gain vector the GPU worker already consumes, so the
binary manifest format and the worker contract are unchanged. Slots are named by index: the
index-to-receptor mapping lives in `resolve_channel` and these two sets make no biological claim
about it. Unity is omitted from the sweep because a single slot at 1.0 is bit-identical to a control.

Defaults request a maximum of 50 deterministic conditions and 100 steps. The
condition cap is filled only with complete three-dose ladders; current catalog
ordering therefore produces 48 rows (three controls plus 15 complete ladders).
`FLY_CONDITIONS` is an upper bound, and both requested and emitted counts are
recorded. For a bounded smoke, set `FLY_CONDITIONS=3 FLY_STEPS=1`; `FLY_DATA_DIR`,
`FLY_CONNECTOME_DIR`, `FLY_SPMV_DIR`, `FLY_GPU_ENGINE`, and `FLY_GPUJOB` can
select local paths. Direct reads are the default. Set `FLY_RAM_STAGE=1` to copy
the five engine inputs into a full-hash, content-addressed tmpfs stage. Rust
checks available capacity, hashes each source while copying, publishes the
stage atomically, verifies reused stages, and records stage identity and timing
in the receipt. `FLY_RAM_STAGE_ROOT` can change the tmpfs cache root.

Each run writes `manifest.bin`, `worker.stdout`, a ZSTD Parquet summary, and
`receipt.json` under `../.work/runs/<run-id>/` (the project's root `.work/runs/`).

## Binary manifest v1

The 20-byte header is `FLYBAT01`, followed by three little-endian `u32` values:
condition count, timestep count, and gain width (must be 8). Each condition is
8 little-endian `f32` values in this order: unknown, ACh, glutamate, GABA,
dopamine, serotonin, octopamine, spare. Unknown is neutral (`1.0`); spare is
zero. State vectors and the graph are shared across the batch, while gains and
state are indexed per condition.

## Evidence boundary

The baseline receipt remains a frozen record of its original run. Its 50 rows
record varied gains but identical neutral telemetry because the old Python
runner did not forward its environment into `gpu-wait` job specs. That runner
now passes `--env` explicitly. The earlier 50-condition release build matched
the corrected Python reference and produced the five-pair benchmark. The current
complete-ladder build emits 48 conditions and has exact five-field telemetry
parity for those rows; see the root `../data/screening_claim_fixes_20260930.md`
and referenced `../.work/runs/` receipt for current-binary evidence. This demonstrates
software parity for the tested workload; it does not validate biological
predictions. The project as a whole is not converted: see the root README and
migration plan for remaining Python tools.
