import unittest

from cooper.ctrnn import Ctrnn
from cooper.technique.walker import Walker
from cooper.util import get_beers_fitness

WALL_TIME = 360


class WalkerTests(unittest.TestCase):
    def test_walker(self):
        seed = 3
        perturbed = Ctrnn()
        perturbed.set_bias(0, 5.154455202973727)
        perturbed.set_bias(1, -10.756384207938911)
        perturbed.set_weight(0, 0, 5.727046375192666)
        perturbed.set_weight(0, 1, 16.0)
        perturbed.set_weight(1, 0, -12.919146270567634)
        perturbed.set_weight(1, 1, 2.397393192433045)
        initial_fitness = get_beers_fitness(perturbed)
        c = Walker(perturbed, seed=seed)
        c.setup()
        while c.time < WALL_TIME:
            c.single_step()
        ctrnn = c.attempts[c.best][0]
        final_fitness = get_beers_fitness(ctrnn)
        self.assertAlmostEqual(initial_fitness, 0.23469221)
        self.assertGreater(final_fitness, 0.6)
