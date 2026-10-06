# C. elegans connectome assets

Hermaphrodite wiring from Cook et al. 2019 (Nature 571:63), SI 5 adjacency matrices,
corrected July 2020, taken from openworm/ConnectomeToolbox. Brain model: see below; no worm
body exists in this repo.

| File (in `data/worm/`, gitignored) | Content |
|---|---|
| `cook2019_si5_corrected_2020-07.xlsx` | source; URL in `tools/build_worm_csr.py` |
| `worm_nodes.txt` | 473 nodes: neurons plus the muscles and end organs the sheets name; line number = CSR index |
| `worm_chemical_csr.bin` | chemical synapses, magic `WRMC`, 4,879 edges, 28,113 synapses |
| `worm_gap_csr.bin` | gap junctions (symmetric sheet), magic `WRMG`, 2,883 edges, 23,313 contacts |

Binary layout matches the fly's `data/fly_connectome_csr.bin` (`build_csr.py`): header
`<III magic, N, nnz>`, `row_ptr int32[N+1]`, `col int32[nnz]` (pre), `weight float32[nnz]`;
row = post, so `y = W x`. Chemical weights are unsigned synapse counts: the source carries
no transmitter sign.

Rebuild: `curl -fL -o data/worm/cook2019_si5_corrected_2020-07.xlsx <URL>` then
`.venv/bin/python tools/build_worm_csr.py` (needs `openpyxl`).
Check: `pytest tests/test_worm_csr.py` rebuilds and compares every CSR edge to the sheet,
checks gap symmetry and the touch/command neurons; skips when the xlsx is absent.

## Brain model (`agent/worm.py`), touch reflex NEGATIVE

Graded conductance model (Wicks 1996, Kunert 2014 parameters) on the 300 neurons with a chemical
output row: 3,669 chemical and 2,174 gap edges. Reversal -48 mV for the 26 GABAergic neurons
(McIntire 1993), 0 mV for every other presynaptic cell. Leak + gap are backward Euler through a
precomputed inverse (gap spectrum reaches 76k/s, explicit Euler would need dt < 26 us); chemical
synapses explicit; dt 0.1 ms. Arms stack as (K, N, N) float32, two batched matmuls per step.
6 arms x 3 columns: 184 us/step, 1.8 s per simulated second, 11.5 MB peak (tracemalloc).

`tools/worm_reflex_check.py`, 1e4 fA into the touch neurons, change in AVA minus AVB vs rest:

| Arm | anterior (want +) | posterior (want -) | pass |
|---|---|---|---|
| real | +0.023 mV | +0.050 mV | no |
| shuffles 1-5 | | | 2 of 5 pass |

Posterior touch drives the backward command like anterior touch does, responses are hundredths of
a mV, and shuffles do as well as the real wiring. The all-excitatory-except-GABA sign rule is the
first suspect: glutamate acts through inhibitory GluCl channels on some command interneurons, and
Wicks 1996 fitted per-synapse signs for this circuit. Tests: `tests/test_worm_brain.py`.

### Cause (`tools/worm_reflex_diag.py`)
1. Chemical overdrive: Cook synapse counts x Gs 100 pS swamp the 10 pS leak, rest sits near -1 mV,
   synaptic activation at 98% of max, so touch input cannot change chemical output; only gap
   junctions carry the stimulus.
2. Gap route favours backward: PLM gaps to AVA/AVD and PVC, and PVC-AVA gaps (up to 22 per pair)
   pull AVA with PVC; PVC-AVB is chemical only.
3. Per-neuron Vth at rest, Gs 100/10/1, Wicks 1996 inhibitory touch->opposite-command synapses and
   forward/backward cross-inhibition: posterior response shrinks to +0.001 mV but never turns
   negative. The reflex is not reproduced by any tested combination; a fitted model (Wicks fitted
   gap and chemical strengths per synapse) or a different readout is needed.

### Fit (`tools/worm_fit.py`), NEGATIVE; blocker located
Fitted one reversal per presynaptic neuron in [-48, 0] mV plus a global chemical scale (antithetic ES,
16 arms, 60 generations, touch only; holdout FLP and ASH). Real graph: loss 0.99 -> 0.95, posterior
+0.21 mV, chemical scale driven to its floor (x0.001), no sign crossed. Shuffles 1-2 fail too. The
real graph "passes" holdout only because FLP and ASH raise AVA before any fit (bias, not wiring).

Blocker: symmetric gap junctions from PLM and PVC onto AVA/AVD. Removing those 16 gap pairs passes
the reflex at chemical scale x1, x0.1 and x0.001 (posterior -0.028 / -0.090 / -0.045 mV). No
reversal choice can undo a symmetric electrical path, so the fit could not succeed. Next: fitted or
rectifying gap strengths (worm UNC-7/UNC-9 junctions can rectify; Wicks fitted gap strengths too).
