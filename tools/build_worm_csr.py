#!/usr/bin/env python3
"""Build CSR matrices of the C. elegans hermaphrodite connectome (Cook et al. 2019).

Source: SI 5 adjacency matrices, corrected July 2020, via openworm/ConnectomeToolbox.
Same binary layout as build_csr.py (row = post, col = pre, y = W * x):
  header <III magic, num_nodes, nnz>, row_ptr int32[N+1], col int32[nnz], weight float32[nnz]
Chemical weights are raw synapse counts (the sheet carries no transmitter sign).
Gap junction weights come from the symmetric sheet, so W is symmetric.
Nodes are neurons plus the muscles and end organs the sheets name, in data/worm/worm_nodes.txt.
"""

import os
import struct

import numpy as np
import openpyxl

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "worm")
XLSX = f"{DIR}/cook2019_si5_corrected_2020-07.xlsx"
URL = ("https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/"
       "SI%205%20Connectome%20adjacency%20matrices%2C%20corrected%20July%202020.xlsx")
SHEETS = {"chemical": ("hermaphrodite chemical", 0x57524D43),  # 'WRMC'
          "gap": ("hermaphrodite gap jn symmetric", 0x57524D47)}  # 'WRMG'


def read_edges(ws):
    """Rows = pre (name in col 3), cols = post (names in row 3), from row/col 4 on."""
    rows = list(ws.iter_rows(values_only=True))
    post = rows[2]
    edges = {}
    for r in rows[3:]:
        if r[2] is None:
            continue
        for j in range(3, len(r)):
            if r[j] and post[j] is not None:
                edges[(r[2].strip(), post[j].strip())] = float(r[j])
    return edges


def read_all(path=XLSX):
    wb = openpyxl.load_workbook(path, read_only=True)
    return {k: read_edges(wb[sheet]) for k, (sheet, _) in SHEETS.items()}


def read_csr(path):
    with open(path, "rb") as f:
        magic, n, nnz = struct.unpack("<III", f.read(12))
        row_ptr = np.frombuffer(f.read(4 * (n + 1)), dtype=np.int32)
        col = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
        w = np.frombuffer(f.read(4 * nnz), dtype=np.float32)
    return magic, row_ptr, col, w


def main():
    all_edges = read_all()
    names = sorted({x for e in all_edges.values() for pair in e for x in pair})
    idx = {s: i for i, s in enumerate(names)}
    with open(f"{DIR}/worm_nodes.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    for kind, (_, magic) in SHEETS.items():
        edges = all_edges[kind]
        pre = np.array([idx[a] for a, _ in edges], dtype=np.int32)
        post = np.array([idx[b] for _, b in edges], dtype=np.int32)
        w = np.array(list(edges.values()), dtype=np.float32)
        order = np.lexsort((pre, post))
        row_ptr = np.zeros(len(names) + 1, dtype=np.int32)
        row_ptr[1:] = np.cumsum(np.bincount(post, minlength=len(names)))
        out = f"{DIR}/worm_{kind}_csr.bin"
        with open(out, "wb") as f:
            f.write(struct.pack("<III", magic, len(names), len(w)))
            f.write(row_ptr.tobytes() + pre[order].tobytes() + w[order].tobytes())
        print(f"{out}: nodes={len(names)} nnz={len(w)} synapses={w.sum():.0f}")


if __name__ == "__main__":
    main()
