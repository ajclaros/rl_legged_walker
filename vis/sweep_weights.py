"""
To run this module, execute `python -m vis.sweep_weights`
    from this repository's base directory.

"""

from typing import List
from typing_extensions import TypeAlias
from multiprocessing import Pool

from jason.ctrnn import CTRNN

from util.fitness_functions import fitness_maximize_output_change

from tqdm.contrib.concurrent import process_map


THREAD_COUNT = 10
BOUNDS = 16.0
STEP = .5

size=2
seed=6
filename = f"data/evolved_solutions/mga_pop-20_gen-1000/ALL/discovery_mga_best_nn{size}_seed-{seed}.json"  




WEIGHT_A = (0, 0)
WEIGHT_B = (1, 0)
LOG_OUTPUT = True
save_filename=f"10sec_fitness_{WEIGHT_A[0]}_{WEIGHT_A[1]}__{WEIGHT_B[0]}_{WEIGHT_B[1]}.csv"


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
    ctrnn = CTRNN( size)
    ctrnn.load_json( filename )
    ctrnn.inner_weights[ WEIGHT_A[0] ][ WEIGHT_A[1] ] = param[0]
    ctrnn.inner_weights[ WEIGHT_B[0] ][ WEIGHT_B[1] ] = param[1]

    fitness = fitness_maximize_output_change(ctrnn, init_duration=0, test_duration=10)
    datum = (param[0], param[1], fitness)

    if LOG_OUTPUT:
        write_to_file(save_filename, f"{datum[0]},{datum[1]},{datum[2]:0.4f}",'a')
    return datum

def write_to_file(save_filename, line, flag='a'):
    with open( save_filename, flag) as filehandle:
        filehandle.write( line+"\n" )
    filehandle.close()


def to_csv(data) :
    header = "a,b,fitness\n"
    lines = [f"{d[0]},{d[1]},{d[2]}" for d in data]
    return header + "\n".join(lines)


if __name__ == "__main__":
    header=f"w{WEIGHT_A[0]}{WEIGHT_A[1]},w{WEIGHT_B[0]}{WEIGHT_B[1]},fitness"
    write_to_file(save_filename, header,'w')
    p = Pool(THREAD_COUNT)
    sweep = get_sweep()
    r = process_map(main, sweep, max_workers=THREAD_COUNT, chunksize=1)
    #data = p.map(main, sweep)
