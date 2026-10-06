"""Do odours that A's learning spilled onto (o3, o4) share more Kenyon-cell drive with A than spared ones (o2, o5)?
Connectivity only, no simulation: KC drive of an odour = excitatory two-hop input ORN -> (any) -> KC, synapse-weighted.
Reported per odour vs A (ODORS6[1]): cosine of KC drive vectors and Jaccard of the top-5% driven KCs.

    .venv/bin/python tools/kc_overlap.py   -> table + PASS/FAIL of the ordering o3,o4 > o2,o5 (exploratory, not pre-registered)
"""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.brain_bench import ODORS6  # noqa: E402
from agent.experiment import cells  # noqa: E402
from agent.verify_shiu import _reference_graph  # noqa: E402

pre, post, w, _ = _reference_graph()
pre, post, w = np.asarray(pre), np.asarray(post), np.asarray(w, float)
c = cells()
n = len(c.klass)
exc = w > 0
W = sparse.csr_matrix((w[exc], (post[exc], pre[exc])), shape=(n, n))  # row = post
kc = c.klass == 'Kenyon_Cell'
drive = []
for o in ODORS6:
    x = np.isin(c.type, ['ORN_' + g for g in o]) & (c.klass == 'olfactory')
    v = W @ (W @ x.astype(float))  # two hops
    drive.append(v[kc])
top = [set(np.argsort(d)[-int(.05 * kc.sum()):]) for d in drive]
A = 1
cos = lambda a, b: float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
spill = {2: .24, 3: .31, 4: .30, 5: .07}  # measured A->o valence gap (LEARNING_SPECIFICITY results)
rows = {o: (cos(drive[A], drive[o]), len(top[A] & top[o]) / len(top[A] | top[o])) for o in (0, 2, 3, 4, 5)}
for o, (cs, jac) in rows.items():
    print(f"o{o} {'-'.join(ODORS6[o]):14} cosine {cs:.3f}  top5% Jaccard {jac:.3f}  spill {spill.get(o, '-')}")
ok = min(rows[3][1], rows[4][1]) > max(rows[2][1], rows[5][1])
r = np.corrcoef([rows[o][1] for o in spill], list(spill.values()))[0, 1]
print(f"Jaccard vs measured spill-over, Pearson r = {r:.2f} (4 points)")
print(('PASS' if ok else 'FAIL'), 'o3,o4 share more top KCs with A than o2,o5 (exploratory)')
