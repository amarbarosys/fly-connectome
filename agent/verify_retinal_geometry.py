"""CPU camera-ray falsifiers and a measured-eye coverage audit. No model fitting."""
import argparse
import json
from pathlib import Path
import numpy as np

from agent.neural_connectome import ROOT, digest
from agent.retinal_geometry import basis, project, ray, snapshot
from agent.prepare_retinal_reference import SOURCE_HASHES
from agent.spatial_vision import MAP, mapping


def neighbour_angles(pq, directions):
    lookup={tuple(row):i for i,row in enumerate(pq)};angles=[]
    for i,(p,q) in enumerate(pq):
        for dp,dq in ((1,0),(0,1),(1,1)):
            j=lookup.get((p+dp,q+dq))
            if j is not None:angles.append(np.degrees(np.arccos(np.clip(directions[i]@directions[j],-1,1))))
    return {'edges':len(angles),'min_median_p95_max_deg':np.percentile(angles,[0,50,95,100]).tolist()}


def marker_gate(arena, camera_id, camera):
    import mujoco
    right,up,forward=basis(camera);records=[]
    # A small near marker also falsifies use of a stereo-offset eye instead of
    # the actual mono projection origin. Render-only geometry replaces the scene.
    for uv in ([.5,.5],[.25,.5],[.75,.5],[.5,.25],[.5,.75]):
        direction=ray(camera,uv);distance=.12/(direction@forward)
        position=np.array(camera['position'])+distance*direction
        arena.eye_renderer.update_scene(arena.sim.mj_data,camera_id,scene_option=arena.eye_options)
        scene=arena.eye_renderer.scene;scene.ngeom=1
        scene.flags[mujoco.mjtRndFlag.mjRND_FOG]=False
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]=False
        # A frontoparallel box has a symmetric image footprint at every bearing.
        mujoco.mjv_initGeom(scene.geoms[0],mujoco.mjtGeom.mjGEOM_BOX,
            np.array([.018,.018,.001]),position,np.column_stack([right,up,forward]).reshape(-1),np.ones(4,np.float32))
        scene.geoms[0].emission=1.
        image=arena.eye_renderer.render();yy,xx=np.nonzero(image.mean(axis=2)>127)
        if not len(xx):raise AssertionError('native marker missing')
        expected=np.asarray(uv)*(np.array(camera['image_size'])-1)
        observed=np.array([xx.mean(),yy.mean()]);error=float(np.max(np.abs(expected-observed)))
        if error>1.:raise AssertionError('native projection error >1 pixel: '+str(error))
        records.append({'uv':uv,'expected_pixel':expected.tolist(),'observed_pixel':observed.tolist(),'max_error_pixels':error})
    return records


def verify(reference, out):
    from agent.neural_single import EthologyArena
    from agent.vision import EyeVision
    out.mkdir(parents=True,exist_ok=False)
    m=mapping();before=digest(MAP)
    with np.load(reference,allow_pickle=False) as archive:
        measured=archive['directions'].copy();pq=archive['pq'].copy()
        manifest=json.loads(str(archive['manifest'].item()))
    declared=json.loads(reference.with_suffix('.json').read_text())
    if declared['artifact_sha256']!=digest(reference) or manifest['source_hashes']!=SOURCE_HASHES:
        raise ValueError('retinal reference identity mismatch')
    if manifest['schema']!='fly-world.retinal-reference.v1' or measured.shape!=(778,3):
        raise ValueError('unexpected retinal reference')
    if not np.isfinite(measured).all() or not np.allclose(np.linalg.norm(measured,axis=1),1,atol=1e-12,rtol=0):
        raise ValueError('invalid measured directions')
    arena=EthologyArena(seed=1);reports=[]
    try:
        vision=EyeVision(spatial=True);vision.register_cameras(arena)
        body=arena.sim.mj_data.xmat[arena.thorax_id].reshape(3,3).copy()
        for eye,camera_id in enumerate(arena.eye_ids):
            camera=snapshot(arena,camera_id)
            # Independent hand oracle: centre axis and 45-degree ray for a 90deg view.
            unit={'forward':[1,0,0],'up':[0,0,1],'near':1.,'bottom':-1.,'top':1.,'center':0.,'image_size':[101,101]}
            assert np.allclose(ray(unit,[.5,.5]),[1,0,0],atol=1e-12)
            assert np.allclose(ray(unit,[1.005,.5]),np.array([1,-1,0])/np.sqrt(2),atol=1e-12)
            uv=np.array([[u,v] for u in (0,.1,.5,.9,1) for v in (0,.2,.5,.8,1)])
            back,visible=project(camera,ray(camera,uv))
            assert visible.all() and np.max(np.abs(uv-back))<1e-12
            _,behind=project(camera,-np.asarray(camera['forward']));assert not behind
            marker=marker_gate(arena,camera_id,camera)
            select=m['eye']==eye
            coords=np.stack([m['u'][select],m['v'][select]],axis=1).astype(float)
            if vision.horizontal_flip[eye]:coords[:,0]=1-coords[:,0]
            if vision.vertical_flip[eye]:coords[:,1]=1-coords[:,1]
            live_rays=ray(camera,coords)@body
            # The left-eye measured reference is absent. Mirroring here is an
            # explicitly hypothetical coverage comparison, never an ID assignment.
            reference_rays=measured.copy()
            if eye==0:reference_rays[:,1]*=-1
            _,reference_visible=project(camera,reference_rays@body.T)
            live_az=np.degrees(np.arctan2(live_rays[:,1],live_rays[:,0]))
            live_el=np.degrees(np.arcsin(np.clip(live_rays[:,2],-1,1)))
            live_pq=np.stack([m['p'][select],m['q'][select]],axis=1)
            unique,indices=np.unique(live_pq,axis=0,return_index=True)
            np.savez_compressed(out/('eye-'+str(eye)+'-rays.npz'),root_ids=m['root_ids'][select],
                                live_body_rays=live_rays,reference_visible=reference_visible)
            reports.append({'eye':'left' if eye==0 else 'right','camera':camera,'marker_records':marker,
                'roundtrip_max_abs_uv':float(np.max(np.abs(uv-back))),
                'live_column_azimuth_range_deg':[float(live_az.min()),float(live_az.max())],
                'live_column_elevation_range_deg':[float(live_el.min()),float(live_el.max())],
                'live_neighbour_angles':neighbour_angles(unique,live_rays[indices]),
                'reference_rays_visible':int(reference_visible.sum()),'reference_rays_total':len(measured),
                'reference_comparison':'hypothetical mirrored right eye' if eye==0 else 'measured right eye',
                'comparison_alignment':'published head axes assumed aligned with native thorax axes',
                'per_neuron_angular_error':'UNVERIFIED: paper Mi1 index to FlyWire root crosswalk missing'})
        assert before==digest(MAP)
        report={'status':'PASS','camera_projection':'PASS','biological_retinal_registration':'UNVERIFIED',
            'reference_sha256':digest(reference),'active_mapping_sha256':before,'reference':manifest,
            'source_sha256':{name:digest(ROOT/'agent'/name) for name in
                             ('retinal_geometry.py','verify_retinal_geometry.py','prepare_retinal_reference.py')},
            'reference_neighbour_angles':neighbour_angles(pq,measured),'eyes':reports,
            'measured_reference_directions':len(measured),
            'right_reference_directions_visible':reports[1]['reference_rays_visible'],
            'right_reference_directions_outside_camera':len(measured)-reports[1]['reference_rays_visible'],
            'max_native_marker_error_pixels':max(row['max_error_pixels'] for eye in reports for row in eye['marker_records']),
            'active_input_changed':False,'gain_fit':False,'calcium_final_used':False,
            'scope':'verified native camera geometry and aggregate reference coverage; no per-neuron biological angular calibration'}
        azimuth=np.degrees(np.arctan2(measured[:,1],measured[:,0]))
        elevation=np.degrees(np.arcsin(measured[:,2]))
        report['measured_right_azimuth_range_deg']=[float(azimuth.min()),float(azimuth.max())]
        report['measured_right_elevation_range_deg']=[float(elevation.min()),float(elevation.max())]
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:arena.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=verify(a.reference,a.out)
    print(json.dumps({k:r[k] for k in ('status','camera_projection','biological_retinal_registration')}))
