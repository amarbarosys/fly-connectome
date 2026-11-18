---
name: fly-connectome
description: Work in fly-connectome, the FlyWire v783 connectome pipeline and embodied neural fly agents. Use before changing the connectome graph, a brain-body link, a pre-registered experiment gate, the screening CLI, or the body-engine app side.
metadata:
  short-description: "fly-connectome: connectome pipeline, embodied fly agents, pre-registered gates"
  owner: amarbaro.org
  license: Apache-2.0
  origin: own
---

# fly-connectome

A whole-brain connectome pipeline (FlyWire v783, 139,255 neurons, 10.5M synapses) with embodied neural fly agents
(MuJoCo, NeuroMechFly body) and a Rust/Mojo screening rebuild. The physics core lives in Kinestride; this repo keeps
the app side.

## Layout

```
agent/           embodied agent system, learning gates (verify_*.py), experiments
agent/body_engine/  live viewer, game and world scenes, check.sh stages; physics core is in Kinestride
fly-screen-rs/   Rust screening CLI (cargo build --release --bin fly-screen)
docs/            CAPABILITIES.md (what runs, what is refused), FLY_ASSET_INDEX.md, plans/, releases/, adr/
tests/           pytest; markers gpu and slow in pytest.ini
```

What the repo can run, and what it refuses, is in `docs/CAPABILITIES.md`. Read it before claiming a capability.

## Checks

- CPU tests: `pytest -m "not gpu"` (markers in `pytest.ini`).
- GPU tests need `-m gpu`. Run them one at a time.
- CPU-only manifest preflight for the screener: `FLY_DRY_ONLY=1 FLY_CONDITIONS=50 FLY_STEPS=100 target/release/fly-screen`.
- Device compile check without a batch: `FLY_GPU_COMPILE_ONLY=1 target/release/fly-screen`.
- Body-engine stages: `agent/body_engine/check.sh`.

## Rules that cost us before

- **Pre-register before the claim run.** Commit the gate and its thresholds, then run the claim. Keep FAIL results in
  history (see `docs/HISTORY.md`, commits `68762a2`, `9c18ab5`, `78a9e1f`).
- **Seed clusters, not mirrored layouts.** Mirrored odor-steering layouts are not independent. Pool p-values per seed
  cluster (correction of 2026-10-02, `docs/releases/v0.1.0.md`).
- **Degree-preserving shuffles are the connectome null.** The in-degree-only random family is a second control.
  Report both (private note `CONNECTOME_BENEFIT`).
- **Raw synapse weights flood the mushroom body.** Normalise per neuron before reading the Kenyon-cell code
  (`docs/releases/v0.5.0.md`).
- **Parity is not biology.** The Rust/Mojo screener matches its Python reference on tested conditions; that is a
  software result (`README.md`, Conversion boundary).
- **Physics core changes go to Kinestride.** The core moved there on 2026-10-09 (`0d0eb56`).

## Porting a stage

Use `skills/body-engine` in Kinestride (github.com/amarbarosys/kinestride). It covers the five-step port, the
float32 tolerance, and the frame conventions.
