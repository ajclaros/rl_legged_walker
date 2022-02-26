from typing import List
from cooper.ctrnn import Ctrnn


class Datum:
    time: float
    """Simulation timestamp for this snapshot"""
    ctrnn: Ctrnn
    """A snapshot of the network at this time"""
    fitness: float
    """Instantaneous fitness during the short evaluation window"""
    distance: float
    """Total distance traveled from start along movement path"""
    displacement: float
    """Displacement from the start (as-the-crow-flies)"""
    best: bool = False
    """If this network's fitness was the highest seen so far"""

    def get_dict(self) -> dict:
        d = {}
        d["time"] = self.time
        d["best"] = self.best
        d["fitness"] = self.fitness
        d["distance"] = self.distance
        d["displacement"] = self.displacement
        for i in range(self.ctrnn.size):
            for j in range(self.ctrnn.size):
                d[f"ctrnn_weights_{i}_{j}"] = self.ctrnn.weights[i, j]
        for i in range(self.ctrnn.size):
            d[f"ctrnn_biases_{i}"] = self.ctrnn.biases[i]
            # d[f"ctrnn_time_constants_{i}"] = self.ctrnn.time_constants[i]
        return d

    def __gt__(self, other: "Datum"):
        return self.fitness > other.fitness


class Run:
    def __init__(self) -> None:
        self.seed: int
        self.initial_ctrnn: Ctrnn
        self.initial_fitness: float
        self.final_ctrnn: Ctrnn
        self.final_fitness: float
        self.history: List[Datum] = []
        self.best: List[Datum] = []

    def log(self, datum: Datum):
        self.history.append(datum)
        if not self.best or datum > self.best[-1]:
            datum.best = True
            self.best.append(datum)

    def to_csv(self, best: bool = False) -> str:
        header = ",".join(self.history[0].get_dict().keys())
        history = self.best if best else self.history
        lines = [",".join(map(str, d.get_dict().values())) for d in history]
        data = "\n".join(lines)
        return header + "\n" + data
