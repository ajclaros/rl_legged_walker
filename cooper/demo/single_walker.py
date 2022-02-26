from cooper.ctrnn import Ctrnn
from cooper.technique.walker import Walker
from cooper.util import get_beers_fitness, get_weight_distance
from cooper.util.run import Datum, Run

WALL_TIME = 360


def run_walker(ctrnn: Ctrnn, seed: int) -> Run:
    run = Run()
    run.initial_ctrnn = ctrnn
    run.initial_fitness = get_beers_fitness(ctrnn)

    c = Walker(ctrnn, seed=seed)
    c.setup()
    while c.time < WALL_TIME:
        c.single_step()
        d = Datum()
        d.time = c.time
        d.ctrnn = c.attempts[-1][0]
        d.fitness = c.attempts[-1][1]
        d.displacement = get_weight_distance(ctrnn, d.ctrnn)
        d.distance = get_weight_distance(c.attempts[-2][0], d.ctrnn)
        run.log(d)

    run.final_ctrnn = c.attempts[c.best][0]
    run.final_fitness = get_beers_fitness(run.final_ctrnn)
    return run


def config_perturbed(ctrnn: Ctrnn) -> int:
    ctrnn.set_bias(0, 5.154455202973727)
    ctrnn.set_bias(1, -10.756384207938911)
    ctrnn.set_weight(0, 0, 5.727046375192666)
    ctrnn.set_weight(0, 1, 16.0)
    ctrnn.set_weight(1, 0, -12.919146270567634)
    ctrnn.set_weight(1, 1, 2.397393192433045)
    return 3


if __name__ == "__main__":
    ctrnn = Ctrnn()
    seed = config_perturbed(ctrnn)
    run = run_walker(ctrnn, seed)
    print(run.to_csv(best=False))
