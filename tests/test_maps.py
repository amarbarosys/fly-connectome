"""Geometry and solvability of the 2D closed-loop maps (CPU, no brain)."""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import maps


def walk(m, follow, steps=150, flies=400, seed=0):
    """Run-and-tumble on the true food fraction (follow) or blind (tumble every step); food bites."""
    rng = np.random.default_rng(seed)
    sign = np.where(np.arange(flies) % 2, 1., -1.)
    walls, sources = maps.mirrored(m, sign)
    pos, head = maps.starts(m, sign, rng)
    last, score = np.full(flies, -1.), np.zeros((flies, 2))
    for _ in range(steps):
        c = maps.concentration(m, pos, sources, sign)
        p = c[:, 0] / np.maximum(c.sum(1), 1e-12) if follow else np.zeros(flies)
        head = np.where(p <= last, head + rng.uniform(-np.pi, np.pi, flies), head)
        last = p
        pos, blocked = maps.move(pos, head, .15, walls)
        head = np.where(blocked, head + rng.uniform(-np.pi, np.pi, flies), head)
        score += maps.eat(pos, sources)
    return score


class TestMaps(unittest.TestCase):
    def test_walls_block_and_flies_stay_inside(self):
        m = maps.build('tmaze')
        walls, _ = maps.mirrored(m, np.ones(1))
        _, blocked = maps.move(np.array([[0., -1.]]), np.array([np.pi]), .5, walls)  # into the stem wall
        self.assertTrue(blocked[0])
        _, blocked = maps.move(np.array([[0., -1.]]), np.array([np.pi / 2]), .15, walls)  # up the stem
        self.assertFalse(blocked[0])

    def test_mirror_swaps_food_and_decoy_sides(self):
        for name in maps.NAMES:
            m = maps.build(name)
            _, s = maps.mirrored(m, np.array([1., -1.]))
            np.testing.assert_allclose(s[0, :, m.axis], -s[1, :, m.axis])

    def test_plume_is_downwind(self):
        m = maps.build('plume')
        _, s = maps.mirrored(m, np.ones(3))
        c = maps.concentration(m, np.array([[2., .35], [2., -.35], [.1, .35]]), s, np.ones(3))
        self.assertGreater(c[0, 0], 5 * c[1, 0])  # in the food plume vs the decoy's line
        self.assertGreater(c[0, 0], c[2, 0])       # downwind beats upwind of the source

    def test_odour_goes_round_walls_not_through(self):
        m = maps.build('maze')
        _, s = maps.mirrored(m, np.ones(2))
        # just below the y=0.4 wall, straight under the food, vs the far end of the same corridor
        c = maps.concentration(m, np.array([[1.2, .3], [-1.2, .1]]), s, np.ones(2))
        self.assertGreater(c[1, 0], c[0, 0])  # the gap side is closer by walking

    def test_food_depletes(self):
        m = maps.build('open')
        _, s = maps.mirrored(m, np.ones(1))
        at = s[0, 0, :2][None]
        bites = sum(maps.eat(at, s)[0, 0] for _ in range(10))
        self.assertEqual(bites, 5)  # a patch holds 5 bites
        self.assertEqual(s[0, 0, 3], 0)

    def test_maps_reward_following_the_odour(self):
        for name in maps.NAMES:
            follow, blind = walk(maps.build(name), True), walk(maps.build(name), False)
            with self.subTest(map=name):
                self.assertGreater(follow[:, 0].sum(), follow[:, 1].sum())     # food over decoy
                self.assertGreater(follow[:, 0].sum(), 1.5 * blind[:, 0].sum())  # beats a blind walker


if __name__ == '__main__':
    unittest.main()
