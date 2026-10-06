import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.work/fly-research/repos/artem-x-meta/fly-arena'))
from fly_arena.environment import Environment
from agent import world_edit


class Model:
    def __init__(self): self.geom_pos = np.zeros((2, 3))
    def geom(self, name): return SimpleNamespace(id=int(name.split('_')[1]))


def world():
    env = Environment([{'id': 'a', 'x': 1., 'y': 2.}, {'id': 'b', 'x': -3., 'y': 0., 'amount': 4.}])
    arena = SimpleNamespace(food_patches=[{'id': 'a', 'x': 1., 'y': 2.}, {'id': 'b', 'x': -3., 'y': 0.}],
                            sim=SimpleNamespace(mj_model=Model()))
    return env, arena


def test_move_updates_odour_contact_and_geom_together():
    env, arena = world()
    probe = (5., 5., .01)
    before = env.odor_at(probe)
    r = world_edit.apply({'food_id': 'b', 'x': 5., 'y': 5.}, env, arena)
    assert r == {'food_id': 'b', 'before': {'x': -3., 'y': 0.}, 'after': {'x': 5., 'y': 5.}}
    assert env.odor_at(probe) > before
    assert arena.food_patches[1]['x'] == 5. and tuple(arena.sim.mj_model.geom_pos[1][:2]) == (5., 5.)
    assert arena.food_patches[0]['x'] == 1.


def test_amount_zero_silences_odour():
    env, arena = world()
    world_edit.apply({'food_id': 'a', 'amount': 0}, env, arena)
    world_edit.apply({'food_id': 'b', 'amount': 0}, env, arena)
    assert env.odor_at((0., 0., .01)) == 0


@pytest.mark.parametrize('edit', [{'food_id': 'zz', 'x': 1}, {'food_id': 'a'}, {'food_id': 'a', 'amount': -1},
                                  {'food_id': 'a', 'x': float('nan')}, {'food_id': 'a', 'x': '3'}, {'food_id': 'a', 'z': 1}])
def test_bad_edit_changes_nothing(edit):
    env, arena = world()
    with pytest.raises(ValueError):
        world_edit.apply(edit, env, arena)
    assert (env.food[0].x, env.food[0].amount, arena.food_patches[0]['x']) == (1., 10., 1.)
