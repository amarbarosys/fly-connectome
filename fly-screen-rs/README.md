# `fly-screen`

Rust owns catalog access, Hill gains, typed run manifests, process orchestration,
hashes, Parquet results, and receipts. `fly_batch_gpu.mojo` owns the resident
GPU simulation. Python remains an independent scientific oracle.

## Build and preflight

```sh
cd /home/<user>/Projects/mojo/fly-spmv
mojo build src/fly_batch_gpu.mojo -o .work/fly_batch_gpu

cd /home/<user>/Projects/science/fly-connectome/fly-screen-rs
cargo test
cargo build --release
FLY_DRY_ONLY=1 FLY_CONDITIONS=3 FLY_STEPS=1 target/release/fly-screen
```

The CPU preflight validates the manifest and full connectome dimensions without
creating a GPU context. Normal runs use `kit/gpujob`, which repeats the dry gate
and submits one GPU job with a 22 GB reservation and a 180 second timeout.

## Run

```sh
target/release/fly-screen
```

Defaults are the first 50 deterministic conditions and 100 steps. For a bounded
smoke, set `FLY_CONDITIONS=3 FLY_STEPS=1..128`; `FLY_DATA_DIR`,
`FLY_CONNECTOME_DIR`, `FLY_SPMV_DIR`, `FLY_GPU_ENGINE`, and `FLY_GPUJOB` can
select local paths. Each run writes `manifest.bin`, `worker.stdout`, a ZSTD
Parquet summary, and `receipt.json` under `../.work/runs/<run-id>/`.

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
now passes `--env` explicitly. Corrected Python and Rust/Mojo telemetry match
exactly for the three acetylcholine dose points, and controls match the frozen
receipt. This demonstrates software parity for the checked conditions; the
global transmitter-gain hypothesis is still not biological validation.
