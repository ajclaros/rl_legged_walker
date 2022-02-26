import numpy as np

from cooper.behavior.oscillator import Oscillator
from cooper.ctrnn import Ctrnn


def get_beers_fitness(ctrnn: Ctrnn, duration: float = 300, window: float = 50) -> float:
    voltages = ctrnn.init_voltage()
    behavior = Oscillator(dt=0.01, size=ctrnn.size, duration=duration, window=window)
    behavior.setup(ctrnn.get_output(voltages))
    while behavior.time < behavior.duration:
        voltages = ctrnn.step(0.01, voltages)
        behavior.grade(ctrnn.get_output(voltages))
    return behavior.fitness


def get_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum(np.power(a + -b, 2)))


def get_weight_distance(a: Ctrnn, b: Ctrnn) -> float:
    return get_distance(a.weights, b.weights)
