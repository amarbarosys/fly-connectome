"""Extract published measured rays without inventing a FlyWire ID crosswalk.

Requires rdata only for this offline conversion; no publication R code is run.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from agent.neural_connectome import digest

COMMIT='99d2a43123db636cedb55af9ff31a59657e7d17e'
SOURCE='https://github.com/reiserlab/eyemap_T4/tree/'+COMMIT
SOURCE_HASHES={
    'data/eyemap.RData':'c3f63c69afdc3f381cdafabb1d376b8e25ec5a71bce67550ae42fcdd79cf618e',
    'data/med_ixy.RData':'4c2430e8e2b8e488d2d26643e141b1622d46b464ea3f43b016d0cb0b27bd5ff7',
    'proc_eyemap.R':'35c0d8d48a67ea6bd11237464f01a6e87f13146e07baef7ad863ec1e8795df6d',
    'eyemap_func.R':'d139cd26ef5aa2ef60b54236b399a1eaef8cc3ce0c95d7026884d34c68d948ac'}


def prepare(source, output):
    import rdata
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('reference already exists; choose a new output')
    for name,expected in SOURCE_HASHES.items():
        if digest(source/name)!=expected:raise ValueError('pinned source identity mismatch: '+name)
    eye=rdata.read_rda(source/'data/eyemap.RData')
    grid=np.asarray(rdata.read_rda(source/'data/med_ixy.RData')['med_ixy'])
    pairs=np.asarray(eye['eyemap']);directions=np.asarray(eye['ucl_rot_sm'])
    if pairs.shape!=(778,2) or directions.shape!=(778,3) or not np.isfinite(pairs).all():
        raise ValueError('unexpected published reference dimensions')
    if not np.array_equal(pairs,pairs.astype(np.int32)):raise ValueError('noninteger source indices')
    pairs=pairs.astype(np.int32)
    if len(np.unique(pairs[:,0]))!=len(pairs) or len(np.unique(pairs[:,1]))!=len(pairs):
        raise ValueError('published mapping is not one-to-one')
    rownames=np.asarray(eye['med_xyz'].coords['dim_0']).astype(np.int32)
    if not np.array_equal(rownames,pairs[:,0]):raise ValueError('medulla row order mismatch')
    lensnames=np.asarray(eye['ucl_rot_sm'].coords['dim_0']).astype(np.int32)
    if not np.array_equal(lensnames,pairs[:,1]):raise ValueError('lens direction row order mismatch')
    if not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12,rtol=0):
        raise ValueError('published rays are not finite unit vectors')
    lookup={int(row[0]):row[1:] for row in grid}
    lattice=np.array([lookup[int(i)] for i in pairs[:,0]],dtype=np.int32)
    if len({tuple(row) for row in lattice})!=len(lattice):raise ValueError('duplicate lattice positions')
    manifest={'schema':'fly-world.retinal-reference.v1','source':SOURCE,
        'source_hashes':dict(SOURCE_HASHES),
        'eye':'right','measured_matched_directions':len(pairs),
        'axes':'x anterior, y left, z dorsal; azimuth atan2(y,x)',
        'identity':'paper Mi1 array indices and lens indices, NOT FlyWire roots or Codex column IDs',
        'auxiliary_extrapolated_points_used':False,
        'left_eye_mirror_measured':False,'active_input_changed':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream,manifest=np.array(json.dumps(manifest)),
                            mi1_index=pairs[:,0],lens_index=pairs[:,1],pq=lattice,directions=directions)
    report=dict(manifest,artifact_sha256=digest(output))
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output)))
