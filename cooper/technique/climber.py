from typing import List, Tuple
import numpy as np

from cooper.ctrnn import Ctrnn
from cooper.technique import Technique


class Climber(Technique):
    def __init__(
        self,
        ctrnn: Ctrnn,
        seed: int = 0,
        mutation: float = 0.05,
        duration: float = 10,
        samples: int = 1,
        dt: float = 0.01,
    ):
        self.seed = seed
        self.progenitor = ctrnn
        self.mutation = mutation
        self.attempts: List[Tuple[Ctrnn, float]] = []
        self.attempt = 0
        self.best = 0
        self.dt = dt
        self.duration = duration
        self.samples = samples
        self.rng = np.random.default_rng(self.seed)
        self.time = 0

    def setup(self):
        voltages = self.progenitor.init_voltage()
        behavior = self.new_behavior(self.progenitor.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = self.progenitor.step(self.dt, voltages)
            behavior.grade(self.progenitor.get_output(voltages))
        self.attempts.append((self.progenitor, behavior.fitness))

    def single_step(self):
        self.attempt += 1
        parent = self.attempts[self.best][0]
        s = lambda: self.sample(parent)
        fitness, best = max([s() for _ in range(self.samples)])
        self.attempts.append((best, fitness))
        if fitness >= self.attempts[self.best][1]:
            self.best = self.attempt
        self.time += self.duration

    def sample(self, parent: Ctrnn) -> Tuple[float, Ctrnn]:
        ctrnn = parent.clone()
        ctrnn.perturb(self.mutation, self.rng)
        voltages = ctrnn.init_voltage()
        behavior = self.new_behavior(ctrnn.get_output(voltages))
        while behavior.time < behavior.duration:
            voltages = ctrnn.step(self.dt, voltages)
            behavior.grade(ctrnn.get_output(voltages))
        return behavior.fitness, ctrnn
