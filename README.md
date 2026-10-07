# Fly connectome project

Whole-brain connectome pipeline and embodied neural fly agents for adult
*Drosophila melanogaster*, built on FlyWire v783 (139,255
neurons, 10.5M synapses) with MuJoCo biomechanics (NeuroMechFly) and a GPU
perturbation kernel on AMD RX 7900 XTX.

Current connectome result: across 32 independent seed clusters (64 mirrored odor-steering layouts),
the real graph did not outperform degree-preserving shuffles (pooled p=0.7551); it outperformed
the in-degree-only random family (pooled p=0.001472). The real arm scored 56/64 and missed its
frozen gate.

Brain dynamics: with raw synapse-count weights, recurrent activity floods the mushroom body (79% of
Kenyon cells active, odour identity lost). Per-neuron normalised gain 35 restores a sparse, distinct
Kenyon-cell code, but mushroom-body output and descending neurons stay silent, so dopamine-gated
learning on that code is not yet demonstrated (`agent.verify_sparse_learning` records FAIL).
See [the matched comparison](docs/plans/CONNECTOME_BENEFIT.md) and the [capability map](docs/CAPABILITIES.md).

Correction 2026-10-02: p-values recomputed with seed clusters (mirrored layouts are not independent); old pooled values 0.6990 (shuffles) and 0.000489 (random) are now 0.7551 and 0.001472; conclusions unchanged: no measured advantage over degree-preserving shuffles, and the real graph remains ahead of the random-column family.

Key areas: five multi-omics data layers, embodied agent system
with checkpoint/restore, and a Rust/Mojo screening rebuild. The compound
screening engine and chemoinformatics database now live in the separate
`fly-pharma` repository. GPU physics on RX 7900 XTX is closed: correct physics was reached after three ROCm-fork fixes, but it did not beat six CPU workers, so CPU MuJoCo remains the backend; our own MuJoCo port is in progress ([body engine](docs/plans/BODY_ENGINE.md)). See [GPU body port status](docs/plans/GPU_BODY_PORT.md) and the
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
- Worm (C. elegans) connectome: [`docs/WORM_ASSET_INDEX.md`](docs/WORM_ASSET_INDEX.md)
- Plans and protocols: [`docs/plans/`](docs/plans/)
- Tests: `pytest tests/` (see `pytest.ini` for markers)
- Toolchain: Mojo 1.1.0 / MAX 26.6.0 in `.venv` (builds go through `~/iTools/bin/mojo-cached`, which uses the `.venv/bin/mojo` of the repo or worktree holding the source)
