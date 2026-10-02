"""Male CNS v1.0 (Janelia FlyEM) as a Shiu-model graph, for G1 on a second animal.

Flat-connectome files from storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
in .work/mcns/. Neurons = bodies with status Traced or Anchor (no glia, orphans, fragments). Edges
between them with at least 5 synapses, the threshold of the FlyWire 783 connection table the
authors' graph is built from. Sign as Shiu: GABA and glutamate inhibitory, everything else
excitatory, from consensus_nt (cell-type level). Classes for the block shuffle = class, else
superclass, as cell_class/super_class on FlyWire.
"""
import numpy as np
import pandas as pd

from agent.neural_connectome import ROOT

DATA = ROOT / '.work/mcns'
RELEASE = 'male CNS v1.0, minconf 0.5, Traced+Anchor, >= 5 synapses'


def graph(threshold=5):
    a = pd.read_feather(DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
                        columns=['bodyId', 'status', 'class', 'superclass', 'type'])
    a = a[a['status'].isin(['Traced', 'Anchor'])].reset_index(drop=True)
    bodies = a['bodyId'].to_numpy(np.int64)
    nt = pd.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather', columns=['body', 'consensus_nt'])
    sign = pd.Series(np.where(nt['consensus_nt'].isin(['gaba', 'glutamate']), -1, 1), index=nt['body'])
    e = pd.read_feather(DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
    e = e[e['weight'].to_numpy() >= threshold]
    order = np.argsort(bodies)
    pre_i, post_i = (np.searchsorted(bodies[order], e[c].to_numpy(np.int64)) for c in ('body_pre', 'body_post'))
    pre_i, post_i = np.minimum(pre_i, len(bodies) - 1), np.minimum(post_i, len(bodies) - 1)
    keep = (bodies[order][pre_i] == e['body_pre'].to_numpy()) & (bodies[order][post_i] == e['body_post'].to_numpy())
    keep &= e['body_pre'].to_numpy() != e['body_post'].to_numpy()
    pre, post = order[pre_i[keep]], order[post_i[keep]]
    w = e['weight'].to_numpy()[keep] * sign.reindex(bodies[pre]).fillna(1).to_numpy()
    names = a['class'].fillna(a['superclass']).fillna('unannotated').to_numpy(object)
    kc = np.flatnonzero(a['class'].to_numpy() == 'Kenyon_Cell').tolist()
    orn = {}
    types = a['type'].fillna('').to_numpy()
    for i in np.flatnonzero((a['class'].to_numpy() == 'olfactory') & pd.Series(types).str.startswith('ORN_').to_numpy()):
        orn.setdefault(types[i][4:], []).append(int(i))
    return pre.astype(np.int64), post.astype(np.int64), w.astype(np.int64), names, (len(bodies), kc, orn)
