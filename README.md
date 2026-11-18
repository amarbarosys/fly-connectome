# fly-connectome

A whole-brain connectome pipeline and embodied neural fly agents for adult *Drosophila melanogaster*. It runs on the
FlyWire v783 connectome (139,255 neurons, 10.5M synapses), uses MuJoCo biomechanics (NeuroMechFly), and runs its brain
and a GPU perturbation kernel (AMD RX 7900 XTX). It is a research record: negative results and FAIL gates stay in the history.

## Results

Matched comparison, private note CONNECTOME_BENEFIT (private note: CONNECTOME_BENEFIT), numbers from
[docs/releases/v0.5.0.md](docs/releases/v0.5.0.md):

- Across 32 independent seed clusters (64 mirrored odor-steering layouts), the real graph did not outperform
  degree-preserving shuffles (pooled p = 0.7551). It did outperform the in-degree-only random family (pooled p = 0.001472).
  The real arm scored 56/64 and missed its frozen gate.
- Correction 2026-10-02: seed clusters replace mirrored layouts as the unit of independence. Old pooled values were
  0.6990 and 0.000489.

Brain dynamics (same file):

- With raw synapse-count weights, recurrent activity floods the mushroom body: 79% of Kenyon cells active, odour
  identity lost.
- With per-neuron normalised gain 35, the Kenyon-cell code is sparse and distinct. Mushroom-body output and descending
  neurons stay silent, so dopamine-gated learning on that code is not yet demonstrated (`agent.verify_sparse_learning`
  records FAIL).

Learning, from commit messages:

-: a 294-label cell-type wiring table, with no original edge kept, recovers the learning gap (5/5).
- `9c18ab5`: the 50 ms eligibility gate FAILs on held-out odours.

Ant central complex (pre-registered gates, each committed before its claim run):

- `68762a2`: A7c FAIL on the real bar, 74/128 home hits.
- `fed8ca9`: A7d calibration reaches 89/128, search FAIL.
- `78a9e1f`: A8 FAIL on seed 101.

Screening (README at `78a9e1f`, Conversion boundary): the Rust/Mojo build matched the corrected Python reference for all
tested conditions at about 24x throughput. These are software parity results, not biological validation.

GPU physics: closed. Correct physics was reached after three ROCm-fork fixes, but it did not beat six CPU workers, so
CPU MuJoCo remains the backend ([docs/releases/v0.5.0.md](docs/releases/v0.5.0.md)).

Full release history: [docs/HISTORY.md](docs/HISTORY.md).

## Quick start

```sh
pytest -m "not gpu"                      # CPU tests; GPU tests are -m gpu (see pytest.ini)

cd fly-screen-rs
cargo build --release --bin fly-screen   # Rust screening CLI
FLY_DRY_ONLY=1 FLY_CONDITIONS=50 FLY_STEPS=100 target/release/fly-screen   # CPU-only manifest preflight
```

A batch run (`target/release/fly-screen` without `FLY_DRY_ONLY`) queues a GPU job. The Mojo worker is built from the
sibling repo `fly-spmv`: `mojo build src/fly_batch_gpu.mojo -o .work/fly_batch_gpu`.

Toolchain: Mojo 1.1.0 and MAX 26.6.0 in `.venv`. Kinestride is installed editable into `.venv`.

## Checks

- `pytest`: `testpaths` are `tests` and `agent/tests` (`pytest.ini`).
- Body-engine stages: `agent/body_engine/check.sh`.
- Device compile check without a batch: `FLY_GPU_COMPILE_ONLY=1 target/release/fly-screen`.
- What runs and what is refused: [docs/CAPABILITIES.md](docs/CAPABILITIES.md).

## What is here

```
agent/              embodied agents, learning gates (verify_*.py), experiments
agent/body_engine/  live viewer, game and world scenes, check.sh stages
fly_arena/ fly_duel/ arena and duel packages, vendored under MIT (LICENSE in fly_arena/)
fly-screen-rs/      Rust screening CLI
docs/               CAPABILITIES.md, FLY_ASSET_INDEX.md, plans/, releases/, adr/, articles/
skills/             agent skill for this repo (fly-connectome)
tests/              pytest
```

Run receipts go to `.work/runs/<run-id>/` and are gitignored.

## Relation to Kinestride

The body engine started here. Its physics core moved to Kinestride, the batched GPU body engine, in commit `0d0eb56`
(2026-10-09), and its agent skill in. Kinestride lives at github.com/amarbarosys/kinestride. This repo keeps
the app side: brain links, the live viewer, game and world scenes, and their `check.sh` stages.

## For coding agents

Read `skills/fly-connectome/SKILL.md` first. For porting a MuJoCo stage, use the `body-engine` skill in Kinestride.

Claude Code: `ln -s "$PWD/skills/fly-connectome" .claude/skills/fly-connectome`.

## Where it is wrong

- The real connectome does not beat degree-preserving shuffles.
- Sparse Kenyon-cell coding is not yet tied to learning.
- The ant and learning gates are FAIL on their real bars at several steps (see [docs/HISTORY.md](docs/HISTORY.md)).
- Screening parity is software parity only.
- Much of the pipeline is still Python (single-compound runner, chemistry database, graph build); the screening Python
  path is kept as the reference oracle (README at `78a9e1f`).

## Data

No FlyWire data is in this repo (`data/` is gitignored). Fetch it yourself and cite it per [codex.flywire.ai](https://codex.flywire.ai):

```sh
tools/fetch_flywire.sh   # annotations (flyconnectome/flywire_annotations @ a83b277) and column assignment
```

The synapse tables and proofread roster have no URL in the code. Get them from https://zenodo.org/records/10676866 into `data/`.
FlyWire's terms apply to all of it; this repo's license does not cover it.

## License

Apache-2.0, see [LICENSE](LICENSE). Portions are translated from MuJoCo (Apache-2.0, DeepMind).
Data sources, such as the FlyWire v783 connectome and its annotations, keep their own terms; this license does not cover them.
