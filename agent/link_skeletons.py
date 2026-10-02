"""Link L2 skeleton nodes to the active root-ID roster: per-neuron morphology, one streaming pass.

Join key is `neuron`, the same FlyWire root ID the roster and the edge table use, so no
crosswalk is involved. Every neuron's nodes are contiguous in the source (139,273 runs for
139,273 unique neurons, verified), so one neuron is buffered at a time and the largest costs
about 5 MB. Coordinates and radii are source nanometres, carried through unconverted.

Cable length sums the distance from each node to its parent. `parent_id` indexes `node_id`
within the same neuron; -1 marks a root. Nodes whose parent is absent are counted, not guessed.
"""
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from agent.neural_connectome import ROOT,digest

SOURCE=ROOT/'data/sk_lod1_783_healed_ds2.parquet'
OUT=ROOT/'.work/agent/skeleton-link'
COLUMNS=['node_id','parent_id','radius','x','y','z','neuron']
BATCH=1<<21
FIELDS=['node_count','cable_length_nm','radius_mean_nm','extent_x_nm','extent_y_nm','extent_z_nm',
    'centroid_x_nm','centroid_y_nm','centroid_z_nm','root_count','absent_parent_count']


def morphology(node_id,parent_id,radius,xyz):
    """Per-neuron metrics. xyz is (n,3) float64 nanometres."""
    linked=parent_id>=0
    order=np.argsort(node_id,kind='stable');sorted_ids=node_id[order]
    slot=np.searchsorted(sorted_ids,parent_id[linked])
    slot=np.clip(slot,0,sorted_ids.size-1)
    found=sorted_ids[slot]==parent_id[linked]
    parent_row=order[slot[found]]
    child=xyz[linked][found];step=child-xyz[parent_row]
    return {'node_count':float(node_id.size),
        'cable_length_nm':float(np.sqrt((step*step).sum(axis=1)).sum()),
        'radius_mean_nm':float(radius.mean()),
        'extent_x_nm':float(np.ptp(xyz[:,0])),'extent_y_nm':float(np.ptp(xyz[:,1])),
        'extent_z_nm':float(np.ptp(xyz[:,2])),
        'centroid_x_nm':float(xyz[:,0].mean()),'centroid_y_nm':float(xyz[:,1].mean()),
        'centroid_z_nm':float(xyz[:,2].mean()),
        'root_count':float((parent_id<0).sum()),
        'absent_parent_count':float((~found).sum())}


def link():
    roster=np.load(ROOT/'data/proofread_root_ids_783.npy')
    index={int(root):i for i,root in enumerate(roster)}
    table={name:np.full(roster.size,np.nan) for name in FIELDS}
    seen=np.zeros(roster.size,bool);skeleton_neurons=0;unrostered=0;rows_read=0
    held=None  # trailing partial neuron carried across a batch boundary

    def flush(block):
        nonlocal skeleton_neurons,unrostered
        skeleton_neurons+=1
        row=index.get(int(block['neuron']))
        if row is None:
            unrostered+=1;return
        metrics=morphology(block['node_id'],block['parent_id'],block['radius'],block['xyz'])
        for name,value in metrics.items():table[name][row]=value
        seen[row]=True

    for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=BATCH,columns=COLUMNS):
        rows_read+=batch.num_rows
        neuron=batch['neuron'].to_numpy(zero_copy_only=False)
        node_id=batch['node_id'].to_numpy(zero_copy_only=False).astype(np.int64)
        parent_id=batch['parent_id'].to_numpy(zero_copy_only=False).astype(np.int64)
        radius=batch['radius'].to_numpy(zero_copy_only=False).astype(np.float64)
        xyz=np.stack([batch[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in 'xyz'],axis=1)
        bounds=np.concatenate(([0],np.flatnonzero(np.diff(neuron)!=0)+1,[neuron.size]))
        for start,stop in zip(bounds[:-1],bounds[1:]):
            block={'neuron':neuron[start],'node_id':node_id[start:stop],
                'parent_id':parent_id[start:stop],'radius':radius[start:stop],'xyz':xyz[start:stop]}
            if held is not None and held['neuron']==block['neuron']:
                block={'neuron':block['neuron'],
                    **{k:np.concatenate([held[k],block[k]]) for k in ('node_id','parent_id','radius','xyz')}}
            elif held is not None:
                flush(held)
            held=block
    if held is not None:flush(held)

    assert rows_read==pq.ParquetFile(SOURCE).metadata.num_rows,'row count mismatch'
    linked=int(seen.sum())
    present=np.isfinite(table['cable_length_nm'])
    report={'status':'PASS','source':str(SOURCE.relative_to(ROOT)),'rows_read':rows_read,
        'roster_neurons':int(roster.size),'skeleton_neurons':skeleton_neurons,
        'linked_neurons':linked,'roster_without_skeleton':int(roster.size-linked),
        'skeleton_without_roster':unrostered,
        'linked_fraction':linked/roster.size,
        'cable_length_nm_total':float(np.nansum(table['cable_length_nm'])),
        'cable_length_nm_median':float(np.nanmedian(table['cable_length_nm'])),
        'node_count_total':float(np.nansum(table['node_count'])),
        'absent_parent_total':float(np.nansum(table['absent_parent_count'])),
        'multi_root_neurons':int(np.nansum(table['root_count']>1)),
        'nonpositive_cable_neurons':int((table['cable_length_nm'][present]<=0).sum()),
        'units':'source nanometres, unconverted','join_key':'neuron == FlyWire root ID == roster entry',
        'scope':'morphology aggregates over the LOD1 ds2 downsampled skeleton, not full-resolution cable length',
        'sources_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            [ROOT/'data/proofread_root_ids_783.npy',Path(__file__)]},
        'recorded_unix':time.time()}
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'morphology.npz',root_id=roster,linked=seen,
        **{name:table[name] for name in FIELDS})
    (OUT/'link.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':print(json.dumps(link(),indent=2))
