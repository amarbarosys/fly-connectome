# Fly connectome project

Whole-brain connectome pipeline, embodied neural fly agents and neuropharmacology
screening for adult *Drosophila melanogaster*, built on FlyWire v783 (139,255
neurons, 10.5M synapses) with MuJoCo biomechanics (NeuroMechFly) and a GPU
perturbation kernel on AMD RX 7900 XTX.

Key areas: five multi-omics data layers, in-silico neuropharmacology engine,
chemoinformatics database (ChEMBL 37 + PubChem 118M), embodied agent system
with checkpoint/restore, and a Rust/Mojo screening rebuild. See
[master plan](docs/plans/FLY_WORLD_MASTER_PLAN.md) for the full program.

## Screening build (Rust/Mojo)

- Rust source and crate: [`fly-screen-rs/`](fly-screen-rs/)
- Mojo resident GPU worker: in sibling repo `fly-spmv/src/fly_batch_gpu.mojo`
- Built local executable: `fly-screen-rs/target/release/fly-screen`
- Run receipts/results: `.work/runs/<run-id>/` (created by the executable, gitignored)
- Migration status and remaining gates: [`docs/plans/SCREENING_REBUILD_PLAN.md`](docs/plans/SCREENING_REBUILD_PLAN.md)

Build the Mojo worker and Rust CLI:

```sh
# From the fly-spmv sibling repo:
mojo build src/fly_batch_gpu.mojo -o .work/fly_batch_gpu

# From this repo:
cd fly-screen-rs
cargo build --release --bin fly-screen
```

Run a batch (queues a GPU job):

```sh
cd fly-screen-rs
target/release/fly-screen
```

CPU-only manifest preflight:

```sh
FLY_DRY_ONLY=1 FLY_CONDITIONS=50 FLY_STEPS=100 target/release/fly-screen
```

For a device-side kernel compile check without running a batch, use
`FLY_GPU_COMPILE_ONLY=1 target/release/fly-screen`; see the Rust README for
the distinction between CPU preflight and GPU compilation.

## Conversion boundary

| Area | Implementation | Status |
|---|---|---|
| Batched screening orchestration, catalog, Hill gains, manifest, Parquet receipt | Rust (`fly-screen-rs`) | Implemented, semantic/schema review open |
| Batched connectome simulation and compact GPU telemetry | Mojo (`fly-spmv`) | Implemented, parity checked |
| Single-compound runner, chemistry DB, graph build, receptor extraction, wet-lab verification, agent experiments | Python | Not converted, some by design |
| Python screening path | Python | Retained as independent oracle/reference |

The Rust/Mojo build matched the corrected Python reference for all tested
conditions at ~24x throughput. These are software parity results, not biological
validation.

## Navigation

- Asset index: [`docs/FLY_ASSET_INDEX.md`](docs/FLY_ASSET_INDEX.md)
- Plans and protocols: [`docs/plans/`](docs/plans/)
- Tests: `pytest tests/` (see `pytest.ini` for markers)
