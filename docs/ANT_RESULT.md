# The fly's central complex steers a model ant home by distance, at 70 %, and only with its real wiring

On 128 held-out outbound paths where ignoring distance is wrong by construction, the model ant points home within
45 deg on 89 of them. Chance is 32. Shuffled wiring under the same calibration reaches 51. Speed held fixed drops it to
3, and with the memory off it scores 0. Both pre-registered bars failed: real needed 96, and the search on shuffled
wiring was allowed at most 41.

The brain is the male CNS (MCNS) central complex (CX), read as a wiring hypothesis for the homologous ant CX. No ant
connectome exists. The model is Shiu LIF, run on the CPU oracle (`agent/ant_cx.py`). Every pre-registration and every
failed arm, A1 to A7d, is in `docs/plans/ANT.md`. The claim form throughout is "a model ant does X".

## The test admits no shortcut

Outbounds are 3 legs of random heading and speed (0.25-1). They are rejection-sampled so the true home lies more than
90 deg from both:
- the speed-blind home (legs summed at equal length);
- "last leg + 180".

Both shortcuts score 0 / 128. On unfiltered paths, a speed-blind integrator already hits 84 %. That flaw made the
earlier no-odometer bars unpassable and hid whether distance was used at all.

## One named mechanism on the full CX does it, and every control falls

The mechanism, M, is Stone 2017's CPU4 rule on PFNd / PFNv. Excitatory input onto a memory cell from EPG, PEG, PEN or
Delta7 is scaled by current speed into a slow level (T 4 s), and the level drives the cell. It is added by hand. It is
the stated hypothesis, not something read off the connectome.

| Gate, 128 held-out paths | Real | No odometer | M off | Shuffled CX | Same search, shuffled CX |
|---|---|---|---|---|---|
| A7c, no learning (seed 92) | 74 | 9 | 0 | 25 | n/a |
| A7d, 32 calibrated gains (seed 96) | **89**, median 30 deg | 3 | 0 | 34 | 51 |

P(X >= 74 | 128, 0.25) = 3e-15; P(X >= 89) = 5e-26.

The search control is the number that weakens the claim. Tuning 32 gains lifts even shuffled wiring 19 hits above
chance, so part of the calibration gain is the optimiser, not the fly. The wiring-specific part is the 38-hit gap
between real and shuffled under the identical search.

## Most of the gain came from fixing the model, not tuning it

| Change | Effect |
|---|---|
| Cut subgraph (556 cells) → full CX (2,950 cells) | The cut held only 16-62 % of the key types' input synapses. Single-leg compass error: median 28 → 11 deg, max 62 → 24. |
| M without PFN → PFN self-feed | The self-feed overshot leg addition (98 vs 45 deg ideal). Without it: 55 → 74 / 128. |
| Calibration (ES on 32 column gains) | 74 → 89 / 128. Search control 51. |
| Earlier: wiring alone, gain search on the cut graph (A4-A5) | No vector memory; at or below the heading-only baseline. |

## What this does not show

- That ant wiring does this. It is fly wiring, plus a hand-added integrator.
- That an ant learns this way. The calibration is an outer evolution-strategy search, not plasticity in the brain.
  A local three-factor rule (A7f) was designed and not run.
- Exact vector addition. Legs still add at 0.5-0.7 of their true weight (`.work/agent/ant/a7-error-diag.log`), with some
  compass lag at leg onset (`a7d-lag-probe.log`).
- A walk home. Outbounds and readout are scripted; the closed-loop row is still open.

## How to break it

Rerun `.venv/bin/python -m agent.ant_cx a7d` on fresh seeds; seeds already used are in `docs/seeds_used.json`.
The claim fails if any of the following happens:
- real falls to the shuffled-search level (about 51 / 128);
- the no-odometer arm climbs above chance;
- a degree-preserving shuffle other than seed 5 matches real.

The decisive open question is whether a local rule inside the brain can reach 89 / 128 while keeping the shuffled
search at chance.
