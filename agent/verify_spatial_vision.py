"""Spatial column input falsifiers, CPU-first with an optional queued GPU parity gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agent.calcium_validation import cue_images
from agent.neural_connectome import CPUBrain, Worker, ROOT, digest
from agent.spatial_vision import MAP, mapping, sample, validate_visual
from agent.vision import EyeVision


def render_gate(out):
    """Controlled render-only marker through actual eye cameras, no physics mutation."""
    import mujoco
    from agent.neural_single import EthologyArena
    arena=EthologyArena(seed=1);m=mapping();records=[]
    try:
        registration=EyeVision(spatial=True);registration.register_cameras(arena)
        for eye,camera in enumerate(arena.eye_ids):
            centroids=[];patterns=[];columns=[]
            rotation=arena.sim.mj_data.cam_xmat[camera].reshape(3,3)
            origin=arena.sim.mj_data.cam_xpos[camera]
            arena.eye_renderer.update_scene(arena.sim.mj_data,camera,scene_option=arena.eye_options)
            projection=arena.eye_renderer.scene.camera[0]
            distance=min(1.,float(projection.frustum_far)*.2)
            half=distance*(projection.frustum_top-projection.frustum_bottom)/(2*projection.frustum_near)
            for offset in (-.5,0.,.5):
                arena.eye_renderer.update_scene(arena.sim.mj_data,camera,scene_option=arena.eye_options)
                scene=arena.eye_renderer.scene;scene.ngeom=1
                scene.flags[mujoco.mjtRndFlag.mjRND_FOG]=False
                scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]=False
                position=origin-distance*rotation[:,2]+offset*half*rotation[:,0]
                mujoco.mjv_initGeom(scene.geoms[0],mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([half*.06,0.,0.]),position,np.eye(3).reshape(-1),np.ones(4,np.float32))
                scene.geoms[0].emission=1.
                image=arena.eye_renderer.render()
                mask=image.mean(axis=2)>127
                assert mask.any(),'marker missing from actual camera'
                centroid=float(np.nonzero(mask)[1].mean());centroids.append(centroid)
                pair=np.zeros((2,*image.shape),np.uint8);pair[eye]=image
                vision=EyeVision(spatial=True)
                vision.horizontal_flip=registration.horizontal_flip;vision.vertical_flip=registration.vertical_flip
                _,_,info=vision.step(pair);vector=vision.spatial_drive
                assert np.any(vector[m['eye']==eye]) and not np.any(vector[m['eye']!=eye])
                patterns.append(info['spatial_drive_sha256'])
                mean=float(np.average(m['u'],weights=vector));columns.append(mean)
                records.append({'eye':eye,'camera_offset':offset,'pixel_x_centroid':centroid,
                                'posterior_column_centroid':mean,'spatial_sha256':patterns[-1],
                                'image_sha256':info['image_sha256'],'pixel_range':[int(image.min()),int(image.max())],
                                'bright_pixels':int(mask.sum()),'frustum_half_at_marker':float(half),
                                'marker_position':position.tolist(),'marker_radius':float(half*.06),
                                'camera_position':origin.tolist(),'render_camera_position':projection.pos.tolist(),
                                'frustum_near':float(projection.frustum_near),'geom_type':int(scene.geoms[0].type),
                                'render_camera_forward':projection.forward.tolist()})
            (out/'render-progress.json').write_text(json.dumps({'records':records,'registration':registration.registration()},indent=2)+'\n')
            assert np.all(np.diff(centroids)>0),'camera image horizontal orientation mismatch: '+str(centroids)
            expected_sign=-1 if registration.horizontal_flip[eye] else 1
            assert np.all(expected_sign*np.diff(columns)>0),'registered posterior axis reversed'
            assert len(set(patterns))==3
        report={'status':'PASS','records':records,'registration':registration.registration(),
                'scope':'native eye projection and registered horizontal spatial ordering; render-only marker, not retinal optics'}
        (out/'render-verification.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:arena.close()


def verify(out, gpu=False, render=False):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();m=mapping()
    graph_sha=digest(ROOT/'data/weights.bin')
    # A bilinear sampler must reproduce a hand-computable affine image exactly.
    yy,xx=np.mgrid[0:17,0:23]
    field=np.stack([xx/22+2*yy/16,10+xx/22+2*yy/16]).astype(np.float32)
    expected=m['u']+2*m['v']+10*m['eye']
    assert np.allclose(sample(field),expected,atol=2e-6,rtol=0)
    flipped=sample(field,(True,False),(False,True))
    expected=np.where(m['eye']==0,1-m['u'],m['u'])+2*np.where(m['eye']==1,1-m['v'],m['v'])+10*m['eye']
    assert np.allclose(flipped,expected,atol=2e-6,rtol=0)
    records=[];vectors=[]
    for cell in (5,10,20):
        angle=(cell+.5)*np.pi/48;image=cue_images(angle,1.,width=48,source_stimulus=True)
        vision=EyeVision(spatial=True);coarse,_,info=vision.step(image)
        vector=vision.spatial_drive.copy();vectors.append(vector)
        assert np.any(vector[m['eye']==0]) and not np.any(vector[m['eye']==1])
        vision.step(image);assert np.array_equal(vector,vision.spatial_drive)
        vision.step(image,blind=True);assert not np.any(vision.spatial_drive)
        records.append({'bearing_rad':float(angle),'coarse':coarse.tolist(),
                        'spatial_sha256':hashlib.sha256(vector.tobytes()).hexdigest(),
                        'nonzero_neurons':int(np.count_nonzero(vector)),'image_sha256':info['image_sha256']})
    assert all(np.array_equal(records[0]['coarse'],row['coarse']) for row in records[1:])
    assert len({row['spatial_sha256'] for row in records})==3
    pair_distance=[float(np.linalg.norm(vectors[i]-vectors[j])) for i,j in ((0,1),(0,2),(1,2))]
    assert min(pair_distance)>0
    # Same-state neural controls: actual mapped drives, no old global visual port.
    brain=CPUBrain();inputs=np.zeros((2,8),np.float32);states=[];counts=[]
    for vector in vectors:
        brain.reset();visual=np.zeros((2,len(vector)),np.float32);visual[0]=vector
        for _ in range(8):reply=brain.step(inputs,visual=visual)
        assert np.isfinite(brain.v).all() and reply['counts'][1]==[0.,0.]
        states.append(np.stack([brain.v[m['nodes'],0],brain.trace[m['nodes'],0]]))
        counts.append(reply['counts'])
    assert all(not np.array_equal(states[i],states[j]) for i,j in ((0,1),(0,2),(1,2)))
    brain.reset();null=brain.step(inputs,visual=np.zeros((2,len(vectors[0])),np.float32))
    assert null['counts']==[[0.,0.],[0.,0.]]
    invalid=np.zeros((2,1),np.float32)
    try:validate_visual(invalid,2)
    except ValueError:pass
    else:raise AssertionError('invalid mapped visual vector accepted')
    dim=EyeVision(spatial=True);dim.step(cue_images(records[0]['bearing_rad'],1/15,width=48,source_stimulus=True))
    ceiling=float(.65*dim.spatial_drive.max()/(1-.85))
    assert ceiling<1,'mapping must not be gain-tuned to force dim first spikes'
    parity=None
    if gpu:
        worker=Worker()
        try:
            for vector in vectors:
                worker.reset();brain.reset();visual=np.zeros((2,len(vector)),np.float32);visual[0]=vector
                expected=brain.step(inputs,visual=visual);actual=worker.step(inputs,visual=visual)
                error=float(np.max(np.abs(np.asarray(expected['features'])-np.asarray(actual['features']))))
                assert error<2e-4 and np.allclose(expected['counts'],actual['counts'],atol=1)
                exported=worker.export_state()
                array_error=max(float(np.max(np.abs(exported[name]-getattr(brain,name)))) for name in ('v','trace'))
                assert array_error<2e-4
                assert np.array_equal(exported['spikes'],brain.spikes) and np.array_equal(exported['ever'],brain.ever)
                parity=max(parity or 0,error,array_error)
        finally:worker.close()
    assert digest(ROOT/'data/weights.bin')==graph_sha
    rendered=render_gate(out) if render else None
    result={'status':'PASS','mapped_neurons':len(m['nodes']),'manifest':m['manifest'],
            'same_coarse_different_spatial_patterns':records,'pair_l2_distances':pair_distance,
            'mapped_neural_states_distinct':True,'finite_state':True,'eye_and_fly_isolation':True,
            'blind_and_zero_controls':True,'static_history_repeatability':True,
            'cold_dim_leaky_ceiling':ceiling,'neural_counts':counts,'graph_weights_unchanged':True,
            'cpu_gpu_full_array_max_abs_error':parity,'gpu_runtime':'PASS' if gpu else 'UNVERIFIED',
            'native_eye_render':rendered,
            'mapping_sha256':digest(MAP),'source_sha256':{name:digest(ROOT/'agent'/name) for name in
                ('spatial_vision.py','vision.py','neural_connectome.py','neural_io.py','neural_worker.mojo','verify_spatial_vision.py')},
            'wall_seconds':time.perf_counter()-started,
            'biological_retinal_registration':'UNVERIFIED','calcium_targets_used':False,
            'scope':'published column identity plus approximate camera sampling; spatial information reaches actual neural state'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'spatial-vectors.npz',vectors=np.stack(vectors),nodes=m['nodes'],root_ids=m['root_ids'])
    print(json.dumps({key:value for key,value in result.items() if key!='manifest'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--render',action='store_true')
    a=p.parse_args();verify(a.out,a.gpu,a.render)
