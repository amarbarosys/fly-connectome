"""Frozen flat-ground arena for the P1 engineering body gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
from agent.neural_single import EthologyArena


SUITE = {
    'schema': 'fly-world.p1-flat.v1', 'mode': 'engineering_agent',
    'seeds': list(range(20)), 'floor_seconds': 60., 'wall_seconds': 5.,
    'dt_s': .01, 'physics_dt_s': .0001, 'upright_projection_threshold': .85,
    'required_upright_fraction_per_seed': .95, 'required_wall_escape_fraction': .9,
    'wall_start_xyz_mm': [19.2, 0., .8], 'wall_start_quaternion': [1., 0., 0., 0.],
    'wall_escape': 'physical east-wall contact then >=1mm inward, heading inward and upright within 5s',
    'terrain': 'flat ground with four boundary walls; stones removed before compilation',
    'controller': 'installed HybridTurningController plus existing Desire contact escape',
    'nominal_drive': [.4, .4], 'neural': False, 'vision': False,
    'self_righting': False, 'climbing': False,
}


class FlatArena(EthologyArena):
    def _configure_world(self):
        super()._configure_world()
        stones = [geom for geom in self.world.ground_geoms if geom.name.startswith('stone_')]
        self.world.ground_geoms[:] = [geom for geom in self.world.ground_geoms if geom not in stones]
        for geom in stones:
            self.world.mjcf_root.delete(geom)
        self.obstacles = []

    def terrain_receipt(self):
        names = [mujoco.mj_id2name(self.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
                 for i in range(self.sim.mj_model.ngeom)]
        assert not any(name.startswith('stone_') for name in names)
        assert all(f'wall_{i}' in names for i in range(4))
        assert not self.obstacles
        return {'stone_geoms': [], 'boundary_walls': [f'wall_{i}' for i in range(4)],
                'model_sha256': hashlib.sha256(self.sim.mj_model.__getstate__()).hexdigest()}


def freeze(output):
    output.mkdir(parents=True, exist_ok=False)
    from agent.neural_connectome import digest
    import fly_arena.arena
    import fly_arena.motor
    from flygym_demo.complex_terrain import HybridTurningController
    import inspect
    paths = [Path(__file__), Path(__file__).with_name('benchmark_body.py'),
             Path(__file__).with_name('desire.py'), Path(__file__).with_name('neural_single.py'),
             Path(fly_arena.arena.__file__), Path(fly_arena.motor.__file__),
             Path(inspect.getfile(HybridTurningController))]
    arena = FlatArena(seed=0)
    try:
        receipt = arena.terrain_receipt()
    finally:
        arena.close()
    manifest = {'suite': SUITE, 'terrain_receipt': receipt,
                'source_sha256': {str(path): digest(path) for path in paths},
                'mujoco_version': mujoco.__version__}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for path in paths:
        destination = output/'sources'/path.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    return manifest
