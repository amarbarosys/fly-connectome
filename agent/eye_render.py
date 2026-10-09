"""Live eyes for the male CNS brain: the engine world rendered from the fly's two eye cameras (body.xml l/r_eye_cam,
fixed on the head), each R1-6 / L1-L3 cell reading the pixel along its own viewing direction (agent/eye_map.py,
azimuth + = left, elevation + = up). The table is built at the model's qpos0, where the thorax frame is the world frame
(x forward, y left, z up) and the head sits at rest; the cameras ride the head, so the cell -> pixel table holds for any pose; per frame the cost is two renders and one gather. Cells outside both camera
frusta read the background. Luminance -> rate as agent/fly_vision.drive.
"""
import mujoco
import numpy as np

from agent import fly_vision as fv

H, W = 48, 64  # render size per eye; ~800 columns per eye, so a few cells share a pixel


class Eyes:
    def __init__(self, m, ids, az, el, mirror=False, renderer=None):
        """mirror: the eye map flipped left/right (mutant: a wall on the left drives the right eye's cells).
        renderer: share one mujoco.Renderer (a second one in the same process rendered a constant image)."""
        self.m, self.d, self.ids = m, mujoco.MjData(m), ids
        self.cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f'fly/{s}_eye_cam_camera') for s in 'lr']
        self.r = renderer or mujoco.Renderer(m, H, W)
        self.gain = fv.eye_gain(ids)
        a, e = np.radians(-az if mirror else az), np.radians(el)
        v = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1)  # head frame
        mujoco.mj_kinematics(m, self.d)
        mujoco.mj_camlight(m, self.d)
        self.pix = np.full(len(ids), -1, np.int64)  # flat index into the stacked (2, H, W) luminance, -1 = unseen
        best = np.full(len(ids), -np.inf)
        for k, c in enumerate(self.cams):
            p = v @ self.d.cam_xmat[c].reshape(3, 3)  # camera frame: x right, y up, looking down -z
            z, t = -p[:, 2], np.tan(np.radians(m.cam_fovy[c]) / 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                x, y = p[:, 0] / z / (t * W / H), p[:, 1] / z / t
            ok = (z > 0) & (abs(x) < 1) & (abs(y) < 1) & (z > best)  # nearest its optical axis wins where the eyes overlap
            col = np.clip(((x + 1) / 2 * W).astype(int), 0, W - 1)
            row = np.clip(((1 - y) / 2 * H).astype(int), 0, H - 1)
            self.pix[ok], best[ok] = k * H * W + row[ok] * W + col[ok], z[ok]
        self.seen = self.pix >= 0
        self.base = None  # per-cell reference luminance (set_base); None = fly_vision's uniform BG

    def set_base(self, q):
        """reference = what each cell sees at pose q (open ground, no obstacle in view): darkening is then measured against
        it, so the ground, darker than BG under both eyes, cancels and only what is not ground drives the cells. Without
        it the ground held the lower field at 120-140 Hz in both eyes and the left-eye gain turned that into a left bias
        (full_fly pilot seed 101: both wall sides turned the fly left)."""
        self.base = np.maximum(self.luminance(q), 1e-3)

    def luminance(self, q):
        """per cell luminance in [0, 1] at body pose q (BG where no camera sees the direction)."""
        d = self.d
        d.qpos[:] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        img = []
        for c in self.cams:
            self.r.update_scene(d, camera=c)
            img.append(self.r.render().astype(np.float32).mean(2) / 255)
        return np.where(self.seen, np.stack(img).ravel()[np.maximum(self.pix, 0)], fv.BG)

    def rates(self, q):
        """per cell Hz: darkening below the background, as fly_vision.drive (BINS levels, left-eye gain)."""
        b = fv.BG if self.base is None else self.base
        dark = np.clip((b - self.luminance(q)) / b, 0, 1)
        return fv.RMAX * np.round(dark * fv.BINS) / fv.BINS * self.gain
