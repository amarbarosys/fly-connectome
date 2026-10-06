# C. elegans connectome assets

Hermaphrodite wiring from Cook et al. 2019 (Nature 571:63), SI 5 adjacency matrices,
corrected July 2020, taken from openworm/ConnectomeToolbox. Connectome only: no worm
brain model or body exists in this repo.

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
