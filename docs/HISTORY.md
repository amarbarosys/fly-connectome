# History

Dates are commit dates from `git log`. Tags mark release notes in `docs/releases/`. The repository has 680 commits.

## Origin (2026-09-30)

- Root commit `ccf4706` adds the FlyWire graph inspection script and schema.
- Eighteen commits in September 2026 build the first pipeline: CSR extraction from FlyWire, a 1:1 virtual fly body
  generator for Blender, a Cycles render, a Rust screener, and a receptor extraction and compound screening CLI.

## Reorganisation (2026-10-02)

- `e114453` moves loose docs into `docs/`; `089067a` consolidates the tests under `pytest.ini`.
- `0c720c3` rewrites the README scope.
- `481a1b7` drops the screening tree; it lives on as its own repository, `fly-pharma`.
- `1d2af31` adds the embodied agent system, its learning gates and schemas.
- `0f1f6cc` adds the fly-world plan set, ADRs and evidence notes under `docs/plans/` and `docs/adr/`.

## Releases

| tag | date | release commit | what it added |
|---|---|---|---|
| v0.1.0 | 2026-10-03 | `bdf92c0` | own GPU body engine (fly and spider, floor and cube, live view); brain settings for every intervention |
| v0.2.0 | 2026-10-03 | `c90a4a4` | live world: fly and spider on endless seeded terrain with landmark cones |
| v0.3.0 | 2026-10-06 | `b287ad4` | fly brain and body linked both ways (vision steering and leg sense); spider pipeline gates |
| v0.4.0 | 2026-10-06 | `89c1599` | play the fly through its neurons; learning specificity experiments; control panel bridge |
| v0.5.0 | 2026-10-07 | `facbf51` | the fly loop on one GPU context; walking straight in the gait kernel; play world and eyes |

Source: `CHANGELOG.md` and `docs/releases/v0.1.0.md` to `v0.5.0.md`.

## Body engine moves out (2026-10-09 to 10-12)

- `79b5057` the body-engine skill moves to Kinestride.
- `0c10868` the physics core moves to Kinestride. `a3f7c1c` points the README at it.
- `b9c238f` the engine GUI uses Kinestride's window; `06e528c` uses its shadow pass.
- `872d766` the engine and renderer buffers come from Kinestride's viewer.

## Experiments after v0.5.0

The later commits are pre-registered tests. Each gate is committed before its claim run, and FAIL results stay in
the history.

- 2026-10-15 to 10-16, the ant central-complex series (A6 to A7). `3be3f7f` reports A7c FAIL at 74/128 home hits. `8c82896` reports A7d FAIL, a calibration search that lifts home hits to 89/128.
  `aafdc8a` reports A7f FAIL.
- 2026-10-21, the learning label ladder. `296ec60` finds the cell-type wiring table carries the learning gap.
- 2026-11-02, the odour-path label arms and the 50 ms eligibility gate. `7df4b0c` reports the table-only arm recovers
  5/5 at 294 labels. `63a2a63` reports the 50 ms eligibility gate FAIL on held-out odours.
- 2026-11-02, `55fb5e9` moves the A8 gate to seed 100 before any brain arm is run on seed 99.
- HEAD at `938df7c` (A8b, seed 101, FAIL A8).
