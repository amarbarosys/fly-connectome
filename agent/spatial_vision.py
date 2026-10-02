"""Exact FlyWire column-ID join, with explicit approximate image registration."""
import argparse
import csv
from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np

from agent.neural_connectome import ROOT, ASSETS, ANNOTATIONS, digest

MAP = ASSETS/'spatial-vision.npz'
SOURCE_URL = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz'
REFERENCE = 'https://github.com/hsseung/OpticLobe.jl/blob/3352e97c37b0f96ab08f27b70b4420f3d4ac2726/src/columnassignment.jl'


def prepare(source, output=MAP):
    output=Path(output)
    if output.exists():raise ValueError('spatial map already exists; choose a new artifact')
    roots=np.load(ROOT/'data/proofread_root_ids_783.npy')
    index={int(root):i for i,root in enumerate(roots)}
    annotations={int(row['root_id']):row for row in csv.DictReader(ANNOTATIONS.open(),delimiter='\t')}
    with gzip.open(source,'rt') as stream:rows=list(csv.DictReader(stream))
    if len({row['root_id'] for row in rows})!=len(rows):raise ValueError('duplicate source root IDs')
    ports=np.load(ASSETS/'ports.npz')['ports']
    chosen=[];excluded=[];bounds={}
    for eye in ('left','right'):
        side=[row for row in rows if row['hemisphere']==eye]
        p=np.array([int(row['p']) for row in side]);q=np.array([int(row['q']) for row in side])
        bounds[eye]=[(q-p).min().item(),(q-p).max().item(),(p+q).min().item(),(p+q).max().item()]
    for row in rows:
        if row['type'] not in ('L1','L2','L3'):continue
        root=int(row['root_id']);node=index.get(root);annotation=annotations.get(root,{})
        eye=row['hemisphere'];side=0 if eye=='left' else 1
        if node is None or annotation.get('cell_type')!=row['type'] or annotation.get('side')!=eye or not ports[node,6+side]:
            excluded.append({'root_id':root,'source_type':row['type'],'local_type':annotation.get('cell_type'),
                             'reason':'missing roster or type/side/port conflict'})
            continue
        p=int(row['p']);q=int(row['q']);h0,h1,v0,v1=bounds[eye]
        chosen.append((node,root,side,p,q,int(row['column_id']),(q-p-h0)/(h1-h0),1-(p+q-v0)/(v1-v0)))
    chosen.sort()
    node,root,side,p,q,column,u,v=map(np.array,zip(*chosen))
    # root IDs must never pass through float64: they exceed its exact integer range.
    root=np.array([item[1] for item in chosen],dtype=np.int64)
    arrays={'nodes':node.astype(np.int32),'root_ids':root,'eye':side.astype(np.int32),
            'p':p.astype(np.int32),'q':q.astype(np.int32),'column_id':column.astype(np.int32),
            'u':u.astype(np.float32),'v':v.astype(np.float32)}
    manifest={'schema':'fly-world.spatial-vision-map.v1','source_url':SOURCE_URL,'reference':REFERENCE,
        'source_sha256':digest(source),'source_rows':len(rows),'mapped_neurons':len(chosen),
        'source_lamina_neurons':len(chosen)+len(excluded),'excluded':excluded,
        'original_lamina_port_neurons':int(ports[:,6:].sum()),
        'unmapped_port_neurons':int(ports[:,6:].sum())-len(chosen),
        'bounds_posterior_dorsal_lattice':bounds,
        'coordinate_units':'dimensionless optic-column lattice; not brain nanometers or retinal degrees',
        'projection':'h=q-p, dorsal=p+q; per-eye full source bounds affinely fit to camera frame',
        'registration_status':'approximate; right-eye axes source-described; homologous left-eye axes assumed',
        'missing_policy':'unmapped lamina neurons receive no natural image drive; no ID interpolation',
        'input_hashes':{str(path.relative_to(ROOT)):digest(path) for path in
                       [ANNOTATIONS,ROOT/'data/proofread_root_ids_783.npy',ASSETS/'ports.npz']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:np.savez_compressed(stream,manifest=np.array(json.dumps(manifest)),**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(manifest,artifact_sha256=digest(output)),indent=2)+'\n')
    return manifest


@lru_cache(maxsize=1)
def mapping():
    with np.load(MAP,allow_pickle=False) as archive:
        arrays={name:archive[name].copy() for name in ('nodes','root_ids','eye','p','q','column_id','u','v')}
        manifest=json.loads(str(archive['manifest'].item()))
    nodes=arrays['nodes'];roots=np.load(ROOT/'data/proofread_root_ids_783.npy')
    if manifest['schema']!='fly-world.spatial-vision-map.v1' or nodes.dtype!=np.int32 or len(np.unique(nodes))!=len(nodes) or np.any((nodes<0)|(nodes>=len(roots))):
        raise ValueError('invalid spatial map node identity')
    if not np.array_equal(roots[nodes],arrays['root_ids']):raise ValueError('spatial map root/roster mismatch')
    if not all(value.shape==nodes.shape for value in arrays.values()):raise ValueError('spatial map dimensions mismatch')
    if np.any((arrays['eye']<0)|(arrays['eye']>1)):raise ValueError('invalid mapped eye')
    for name in ('u','v'):
        if not np.isfinite(arrays[name]).all() or np.any((arrays[name]<0)|(arrays[name]>1)):raise ValueError('invalid image coordinate')
    for relative,expected in manifest['input_hashes'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('spatial map input identity changed: '+relative)
    arrays['manifest']=manifest;arrays['sha256']=digest(MAP)
    return arrays


def validate_visual(values, flies):
    values=np.asarray(values,dtype=np.float32)
    if values.shape!=(flies,len(mapping()['nodes'])) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('visual values must match mapped neurons and lie in [0,1]')
    return values


def sample(image, horizontal_flip=(False,False), vertical_flip=(False,False)):
    m=mapping();eye=m['eye'];u=m['u'].copy();v=m['v'].copy()
    u=np.where(np.asarray(horizontal_flip)[eye],1-u,u);v=np.where(np.asarray(vertical_flip)[eye],1-v,v)
    x=u*(image.shape[2]-1);y=v*(image.shape[1]-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,image.shape[2]-1);y1=np.minimum(y0+1,image.shape[1]-1)
    dx=x-x0;dy=y-y0
    return ((1-dy)*((1-dx)*image[eye,y0,x0]+dx*image[eye,y0,x1])+
            dy*((1-dx)*image[eye,y1,x0]+dx*image[eye,y1,x1])).astype(np.float32)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=MAP);args=parser.parse_args()
    result=prepare(args.source,args.output)
    print(json.dumps({key:result[key] for key in ('mapped_neurons','source_rows','unmapped_port_neurons')}))
