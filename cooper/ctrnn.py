from typing_extensions import TypeAlias
from copy import deepcopy
import numpy as np

from util.activation import sigmoid
from util.range import Range

Voltage: TypeAlias = np.ndarray


class Ranges:
    weights: Range = Range(-16, 16)
    biases: Range = Range(-16, 16)
    time_constants: Range = Range(0.5, 10)


class Ctrnn:
    def __init__(self, size: int = 2) -> None:
        self.size = size
        self.biases = np.zeros(self.size)
        self.time_constants = np.ones(self.size)
        self._inv_time_constants = 1.0 / self.time_constants
        self.weights = np.zeros((self.size, self.size))

    def init_voltage(self) -> Voltage:
        """Create a new voltage instance where all neurons have 0V"""
        return np.zeros(self.size)

    def set_bias(self, neuron: int, bias: float) -> None:
        """Set an individual neuron's bias"""
        self.biases[neuron] = Ranges.biases.clip(bias)

    def set_time_constant(self, neuron: int, time_constant: float) -> None:
        """Set an individual neuron's time constant"""
        self.time_constants[neuron] = Ranges.time_constants.clip(time_constant)
        self._inv_time_constants[neuron] = 1.0 / self.time_constants[neuron]

    def set_weight(self, pre: int, post: int, weight: float) -> None:
        """Set the synaptic weight from a presynaptic to a postsynaptic neuron"""
        self.weights[pre][post] = Ranges.weights.clip(weight)

    def perturb(
        self,
        change: float,
        rng: np.random.Generator,
        weights: bool = True,
        biases: bool = True,
    ) -> None:
        """Slightly modify the synaptic weights and biases of this network"""
        if weights:
            range = Ranges.weights
            direction = rng.uniform(-1, 1, size=(self.size, self.size))
            magnitude = np.sqrt(np.sum(np.power(direction.flat, 2))) or 1
            direction *= rng.uniform(0, change * range.max) / magnitude
            self.weights = (self.weights + direction).clip(range.min, range.max)
        if biases:
            range = Ranges.biases
            direction = rng.uniform(-1, 1, size=self.size)
            magnitude = np.sqrt(np.sum(np.power(direction.flat, 2))) or 1
            direction *= rng.uniform(0, change * range.max) / magnitude
            self.biases = (self.biases + direction).clip(range.min, range.max)

    def get_output(self, voltages: Voltage) -> np.ndarray:
        """Convert voltage values to 0-1 using sigmoid activation"""
        return sigmoid(voltages + self.biases)

    def step(self, dt: float, voltages: Voltage, inputs: Voltage = None) -> Voltage:
        """Increment voltages by a single timestep"""
        inputs = inputs if inputs != None else np.zeros(self.size)
        net = inputs + np.dot(self.weights.T, sigmoid(voltages + self.biases))
        return voltages + dt * (self._inv_time_constants * (-voltages + net))

    def clone(self) -> "Ctrnn":
        """Create a copy of this network"""
        return deepcopy(self)
