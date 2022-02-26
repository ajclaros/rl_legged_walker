"""
To run this module, execute `python -m cooper.demo.sweep_weights`
    from this repository's base directory.

To save the outputs to a file, use stdout redirection:
    `python -m cooper.demo.sweep_weights > data.csv`
"""

from typing import List
from typing_extensions import TypeAlias
from multiprocessing import Pool

from cooper.ctrnn import Ctrnn
from cooper.util import get_beers_fitness

from tqdm.contrib.concurrent import process_map

# Weight: TypeAlias = tuple[int, int]
# Datum: TypeAlias = tuple[float, float, float]
# Param: TypeAlias = tuple[float, float]

THREAD_COUNT = 10
BOUNDS = 16.0
STEP = 4
PROGENITOR = Ctrnn()
PROGENITOR.set_bias(0, 4.515263949538321)
PROGENITOR.set_bias(1, -9.424874214362415)
PROGENITOR.set_weight(0, 0, 5.803844919954994)
PROGENITOR.set_weight(0, 1, 16.0)
PROGENITOR.set_weight(1, 0, -16.0)
PROGENITOR.set_weight(1, 1, 3.5073044750632754)
WEIGHT_A = (0, 0)
WEIGHT_B = (1, 1)
LOG_OUTPUT = True


def get_sweep(step: float = STEP, bounds: float = BOUNDS) :
    params=  []
    y = -bounds
    while y <= BOUNDS:
        x = -bounds
        while x <= BOUNDS:
            params.append((x, y))
            x += step
        y += step
    return params


def main(param):
    ctrnn = PROGENITOR.clone()
    ctrnn.weights[WEIGHT_A] = param[0]
    ctrnn.weights[WEIGHT_B] = param[1]
    fitness = get_beers_fitness(ctrnn)
    datum = (param[0], param[1], fitness)
    if LOG_OUTPUT:
        print(f"{datum[0]},{datum[1]},{datum[2]:0.4f}")
    return datum


def to_csv(data) :
    header = "a,b,fitness\n"
    lines = [f"{d[0]},{d[1]},{d[2]}" for d in data]
    return header + "\n".join(lines)


if __name__ == "__main__":
    print(f"w{WEIGHT_A[0]}{WEIGHT_A[1]},w{WEIGHT_B[0]}{WEIGHT_B[1]},fitness", flush=True)
    p = Pool(THREAD_COUNT)
    sweep = get_sweep()
    r = process_map(main, sweep, max_workers=THREAD_COUNT, chunksize=1)
    #data = p.map(main, sweep)
