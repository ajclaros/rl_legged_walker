import leggedwalker
import numpy as np
from rl_ctrnn import RL_CTRNN
from ctrnn import CTRNN
from walking_task import WalkingTask
import datetime
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path

def learn(duration, size, windowsize, init_flux, max_flux,
          min_period, max_period, conv_rate, learn_rate,
          bias_init_flux, bias_max_flux, bias_min_period,
          bias_max_period, bias_conv_rate, starting_genome, log_data,
          verbose, generator_type='RPG', configuration=[0], prob=0.0):

    learner = WalkingTask(
            duration=duration,
            size=size,
            stepsize=0.1,
            running_window_mode=True,
            running_window_size=windowsize,
            init_flux_amp= init_flux,
            max_flux_amp=40,
            flux_period_min=min_period,
            flux_period_max=max_period,
            flux_conv_rate=conv_rate,
            learn_rate=learn_rate,
            bias_init_flux_amp=bias_init_flux,
            bias_max_flux_amp=bias_max_flux,
            bias_flux_period_min=min_period,
            bias_flux_period_max=max_period,
            bias_flux_conv_rate=conv_rate,
        )
    weights = starting_genome[0 : size* size]
    learner.setWeights(weights.reshape((size, size)))
    learner.setBiases(starting_genome[size* size: size* size+ size])
    learner.setTimeConstants(starting_genome[size* size+size:])
    learner.initializeState(np.zeros(size))
    body = leggedwalker.LeggedAgent()
    if log_data:
        datalogger = DataLogger()
        learner.simulate(
            body,
            learning_start=windowsize,
            datalogger=datalogger,
            verbose=verbose,
            generator_type=generator_type,
            configuration=configuration,
            prob=prob
        )
    else:
        learner.simulate(body, learning_start=4000, verbose=verbose)

    end_fitness = fitnessFunction(learner.recoverParameters())

    if verbose>0:
        print(f"endFitness: {end_fitness}")
    if log_data:
        # if data is being saves, save the end fiteness
        datalogger.data["end_fitness"] = end_fitness
        filepath = Path(f"./data/end_fit-{int(np.round(end_fitness,5)*100000)}")
        datalogger.save(filepath)
