import numpy as np

from cooper.behavior.oscillator import Oscillator
from cooper.ctrnn import Ctrnn


class Technique:
    def __init__(
        self, ctrnn: Ctrnn, seed: int = 0, dt: float = 0.05, duration: float = 10
    ) -> None:
        self.ctrnn = ctrnn
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.duration = duration
        self.dt = dt

    def new_behavior(self, state: np.ndarray) -> Oscillator:
        b = Oscillator(self.dt, duration=self.duration, window=self.duration)
        b.setup(state)
        return b
