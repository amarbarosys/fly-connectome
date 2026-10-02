"""Legacy image summaries and optional column-resolved visual drive."""
import hashlib
import numpy as np


class EyeVision:
    def __init__(self, *, spatial=False):
        self.previous=None;self.spatial=spatial;self.spatial_drive=None
        self.horizontal_flip=(False,False);self.vertical_flip=(False,False)
        self.camera_axes=None
        if spatial:
            from agent.spatial_vision import mapping
            self.mapping=mapping()

    def register_cameras(self, arena):
        """Orient image axes against body forward/up; optical calibration is separate."""
        body=arena.sim.mj_data.xmat[arena.thorax_id].reshape(3,3)
        horizontal=[];vertical=[];axes=[]
        for camera in arena.eye_ids:
            rotation=arena.sim.mj_data.cam_xmat[camera].reshape(3,3)
            forward=float(rotation[:,0]@body[:,0]);up=float(rotation[:,1]@body[:,2])
            if abs(forward)<1e-6 or abs(up)<1e-6:raise ValueError('ambiguous camera/body axis registration')
            horizontal.append(forward>0);vertical.append(up<0)
            axes.append({'image_right_dot_body_forward':forward,'image_up_dot_body_up':up})
        self.horizontal_flip=tuple(horizontal);self.vertical_flip=tuple(vertical);self.camera_axes=axes

    def registration(self):
        return {'spatial':self.spatial,'horizontal_flip':list(self.horizontal_flip),
                'vertical_flip':list(self.vertical_flip),'camera_axes':self.camera_axes,
                'mapping_sha256':self.mapping['sha256'] if self.spatial else None}

    def step(self, eyes, *, blind=False):
        pixels=np.asarray(eyes)
        if pixels.ndim!=4 or pixels.shape[0]!=2 or pixels.shape[-1]!=3 or pixels.dtype!=np.uint8:
            raise ValueError('vision requires two uint8 RGB eye images')
        gray=np.tensordot(pixels.astype(np.float32)/255.,np.array([.2126,.7152,.0722],np.float32),axes=(-1,0))
        old_previous=self.previous
        brightness=gray.mean(axis=(1,2));contrast=gray.std(axis=(1,2))
        motion=np.zeros(2) if self.previous is None else np.abs(gray-self.previous).mean(axis=(1,2))
        darkening=np.zeros(2) if self.previous is None else np.maximum(self.previous-gray,0).mean(axis=(1,2))
        self.previous=gray.copy()
        if self.spatial:
            from scipy.ndimage import uniform_filter
            from agent.spatial_vision import sample
            mean=uniform_filter(gray,size=(1,3,3),mode='nearest')
            variance=np.maximum(uniform_filter(gray*gray,size=(1,3,3),mode='nearest')-mean*mean,0)
            # Same brightness/contrast/motion coefficients as the legacy summary,
            # evaluated locally. Three-pixel patches are an engineering choice.
            local_motion=np.zeros_like(gray) if old_previous is None else np.abs(gray-old_previous)
            field=np.clip(.35*mean+1.5*np.sqrt(variance)+2*local_motion,0,1)
            self.spatial_drive=sample(field,self.horizontal_flip,self.vertical_flip)
            if blind:self.spatial_drive[:]=0
        # Static contrast/brightness and frame change supply lamina ports;
        # darkening is a coarse threat proxy, not a validated looming detector.
        visual=np.clip(.35*brightness+1.5*contrast+2*motion,0,1)
        threat=np.clip(6*darkening,0,1)
        if blind:visual[:]=0;threat[:]=0
        info={'source':'rendered left/right eye RGB','blind':bool(blind),'shape':list(pixels.shape),
              'brightness':brightness.tolist(),'contrast':contrast.tolist(),'motion':motion.tolist(),
              'darkening':darkening.tolist(),'visual_drive':visual.tolist(),'threat_proxy':threat.tolist(),
              'image_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
        if self.spatial:
            info.update({'encoding':'column-resolved approximate camera registration',
                'spatial_mapping_sha256':self.mapping['sha256'],'spatial_neurons':len(self.spatial_drive),
                'spatial_drive_sha256':hashlib.sha256(self.spatial_drive.tobytes()).hexdigest(),
                'spatial_drive_nonzero':int(np.count_nonzero(self.spatial_drive)),
                'spatial_drive_mean':float(self.spatial_drive.mean()),'spatial_drive_max':float(self.spatial_drive.max()),
                'camera_registration':self.registration()})
        return visual,threat,info
