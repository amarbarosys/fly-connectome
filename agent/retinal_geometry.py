"""Perspective ray geometry for auditing the eye renderer, not biological optics."""
import numpy as np


def basis(camera):
    forward=np.asarray(camera['forward'],dtype=float)
    up=np.asarray(camera['up'],dtype=float)
    forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,up);right=right/np.linalg.norm(right)
    up=np.cross(right,forward)
    return right,up,forward


def ray(camera, uv):
    """Top-left image coordinates, 0/1 at the first/last pixel centres."""
    uv=np.asarray(uv,dtype=float)
    if uv.shape[-1]!=2 or not np.isfinite(uv).all():raise ValueError('invalid image coordinates')
    right,up,forward=basis(camera)
    width,height=camera['image_size']
    screen=(uv*np.array([width-1,height-1])+.5)/[width,height]
    bottom,top=camera['bottom'],camera['top'];span=(top-bottom)*width/height
    x=camera['center']+(screen[...,0]-.5)*span
    y=top-screen[...,1]*(top-bottom)
    vectors=x[...,None]*right+y[...,None]*up+camera['near']*forward
    return vectors/np.linalg.norm(vectors,axis=-1,keepdims=True)


def project(camera, directions):
    """Return unclipped image coordinates and a geometric visibility mask."""
    directions=np.asarray(directions,dtype=float)
    if directions.shape[-1]!=3 or not np.isfinite(directions).all():raise ValueError('invalid directions')
    right,up,forward=basis(camera);depth=directions@forward
    with np.errstate(divide='ignore',invalid='ignore'):
        x=(directions@right)*camera['near']/depth
        y=(directions@up)*camera['near']/depth
    width,height=camera['image_size'];span=(camera['top']-camera['bottom'])*width/height
    screen=np.stack([.5+(x-camera['center'])/span,
                     (camera['top']-y)/(camera['top']-camera['bottom'])],axis=-1)
    uv=(screen*[width,height]-.5)/[width-1,height-1]
    visible=(depth>0)&np.isfinite(screen).all(axis=-1)&(screen>=0).all(axis=-1)&(screen<=1).all(axis=-1)
    return uv,visible


def snapshot(arena, camera_id):
    """Read the mono renderer's averaged stereo camera, confirmed by marker tests."""
    import mujoco
    arena.eye_renderer.update_scene(arena.sim.mj_data,camera_id,scene_option=arena.eye_options)
    if arena.eye_renderer.scene.stereo!=0:raise ValueError('retinal audit requires mono rendering')
    camera=mujoco.mjv_averageCamera(*arena.eye_renderer.scene.camera)
    if camera.orthographic:raise ValueError('retinal audit requires perspective rendering')
    return {'position':camera.pos.tolist(),'forward':camera.forward.tolist(),'up':camera.up.tolist(),
            'near':float(camera.frustum_near),'far':float(camera.frustum_far),
            'bottom':float(camera.frustum_bottom),'top':float(camera.frustum_top),
            'center':float(camera.frustum_center),
            'image_size':[arena.eye_renderer.width,arena.eye_renderer.height],
            'fovy_deg':float(arena.sim.mj_model.cam_fovy[camera_id])}
