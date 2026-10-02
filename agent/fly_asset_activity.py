"""Bounded spatial annotation of recorded neuron IDs using local fly assets."""
import csv
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ROOT=Path('.work/agent/skillopt-pilot')


def prepare():
    reader=ipc.open_file(pa.memory_map('data/flywire_synapses_783.feather','r'))
    columns=['id','pre_pt_root_id','post_pt_root_id',*[f'{side}_pt_position_{axis}' for side in ('pre','post') for axis in 'xyz']]
    batch=reader.get_batch(0).slice(0,8192)
    table=pa.Table.from_batches([batch]).select(columns)
    np.savez(ROOT/'synapse-prefix.npz',**{name:table[name].to_numpy() for name in columns})
    skeleton=pq.ParquetFile('data/sk_lod1_783_healed_ds2.parquet')
    batch=next(skeleton.iter_batches(batch_size=65536,columns=['neuron','node_id','parent_id','x','y','z']))
    np.savez(ROOT/'skeleton-prefix.npz',**{name:batch.column(i).to_numpy() for i,name in enumerate(batch.schema.names)})
    metadata=skeleton.schema_arrow.metadata or {}
    unit_values={value.decode() for key,value in metadata.items() if key.endswith(b':units')}
    x={'sources':[{'path':str(Path(p).resolve()),'size_bytes':Path(p).stat().st_size} for p in
        ['data/fca_head.h5ad','data/flywire_synapses_783.feather','data/sk_lod1_783_healed_ds2.parquet','data/sub-13_ses-20201201_behavior_ophys.nwb']],
       'synapse_sample_rows':len(table),'synapse_sample':'first 8192 rows of first Arrow record batch',
       'synapse_position_units':'UNVERIFIED raw coordinate frame; no scaling or skeleton alignment applied',
       'skeleton_sample_rows':batch.num_rows,'skeleton_sample':'first 65536 node rows',
       'skeleton_total_rows':skeleton.metadata.num_rows,'skeleton_neuron_unit_metadata_values':sorted(unit_values),
       'id_join':'exact int64 FlyWire root IDs, never row order; partial sampled coverage',
       'excluded':'FCA and NWB are inventoried only; no documented per-neuron crosswalk or HDF5 reader available'}
    (ROOT/'asset-provenance.json').write_text(json.dumps(x,indent=2)+'\n')
    print(json.dumps({k:x[k] for k in ['synapse_sample_rows','skeleton_sample_rows','skeleton_total_rows','skeleton_neuron_unit_metadata_values']}))


def join():
    with (ROOT/'session/touched-neurons.csv').open() as stream:
        records={int(row['flywire_root_id']):row for row in csv.DictReader(stream)}
    synapses=np.load(ROOT/'synapse-prefix.npz');skeleton=np.load(ROOT/'skeleton-prefix.npz')
    def labels(roots):
        return np.asarray([int(records.get(int(root),{}).get('fired',0)) for root in roots],np.uint8)
    pre,post=synapses['pre_pt_root_id'],synapses['post_pt_root_id']
    neurons=skeleton['neuron']
    roster=set(map(int,np.load('data/proofread_root_ids_783.npy')))
    pre_match=np.asarray([int(root) in roster for root in pre]);post_match=np.asarray([int(root) in roster for root in post])
    skeleton_match=np.asarray([int(root) in roster for root in neurons])
    pre_fired,post_fired,node_fired=labels(pre),labels(post),labels(neurons)
    np.savez_compressed(ROOT/'session/spatial-activity.npz',
        synapse_ids=synapses['id'],pre_root_ids=pre,post_root_ids=post,pre_roster_match=pre_match,post_roster_match=post_match,
        pre_ever_fired=pre_fired,post_ever_fired=post_fired,
        pre_positions=np.column_stack([synapses[f'pre_pt_position_{a}'] for a in 'xyz']),
        post_positions=np.column_stack([synapses[f'post_pt_position_{a}'] for a in 'xyz']),
        skeleton_root_ids=neurons,skeleton_node_ids=skeleton['node_id'],skeleton_parent_ids=skeleton['parent_id'],
        skeleton_roster_match=skeleton_match,skeleton_ever_fired=node_fired,
        skeleton_positions_nm=np.column_stack([skeleton[a] for a in 'xyz']))
    report={'status':'PASS','sampled_synapses':len(pre),'pre_roster_matches':int(pre_match.sum()),
        'post_roster_matches':int(post_match.sum()),'synapses_with_recorded_fired_endpoint':int(np.count_nonzero(pre_fired|post_fired)),
        'sampled_skeleton_nodes':len(neurons),'skeleton_nodes_in_roster':int(skeleton_match.sum()),
        'sampled_skeleton_neurons':len(np.unique(neurons)),
        'sampled_skeleton_neurons_recorded_fired':len(np.unique(neurons[node_fired.astype(bool)])),
        'coordinate_alignment':'not attempted; skeleton nm metadata, synapse frame unverified',
        'coverage':'bounded prefix samples, not full-dataset coverage'}
    (ROOT/'session/spatial-activity-metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','join'));a=p.parse_args()
    prepare() if a.stage=='prepare' else join()
