import leggedwalker
import numpy as np
from rl_ctrnn import RL_CTRNN
from ctrnn import CTRNN
from walking_task import WalkingTask
import datetime
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path
import os

def learn(starting_genome, duration=2000, size=2, windowsize=4000, stepsize=0.1, init_flux=2.75, max_flux=10,
          min_period=300, max_period=400, conv_rate=0.004, learn_rate=0.004,
          bias_init_flux=2.75, bias_max_flux=10, bias_min_period=300,
          bias_max_period=400, bias_conv_rate=0.004, log_data=False,
          verbose=1.00, generator_type='RPG', configuration=[0], tolerance=None, tracking_parameters=None, filename = None, folderName=None,
          track_fitness=False):

    learner = WalkingTask(
            duration=duration,
            size=size,
            stepsize=stepsize,
            running_window_mode=True,
            running_window_size=windowsize,
            init_flux_amp= init_flux,
            max_flux_amp=max_flux,
            flux_period_min=min_period,
            flux_period_max=max_period,
            flux_conv_rate=conv_rate,
            learn_rate=learn_rate,
            bias_init_flux_amp=bias_init_flux,
            bias_max_flux_amp=bias_max_flux,
            bias_flux_period_min=min_period,
            bias_flux_period_max=max_period,
            bias_flux_conv_rate=conv_rate
        )
    weights = starting_genome[0 : size* size]
    learner.setWeights(weights.reshape((size, size)))
    learner.setBiases(starting_genome[size* size: size* size+ size])
    learner.setTimeConstants(starting_genome[size* size+size:])
    learner.initializeState(np.zeros(size))
    body = leggedwalker.LeggedAgent()
    if log_data:
        datalogger = DataLogger()
        for var in tracking_parameters:
            if "weight" in var:
                datalogger.data[var] = np.zeros(
                    (int(duration/stepsize), size, size)
                )

            elif "bias" in var or "voltage" in var or "outputs" in var:
                datalogger.data[var] = np.zeros((int(duration/stepsize), size))
            else:
                datalogger.data[var] = np.zeros(int(duration/stepsize))
        learner.simulate(
            body,
            learning_start=windowsize,
            datalogger=datalogger,
            verbose=verbose,
            generator_type=generator_type,
            configuration=configuration,
            tolerance=tolerance,
            track = track_fitness
        )
    else:
        learner.simulate(body, learning_start=4000, verbose=verbose)

    start_fitness = fitnessFunction(starting_genome, N=size, generator_type=generator_type, configuration=configuration)
    end_fitness = fitnessFunction(learner.recoverParameters(), N=size, generator_type=generator_type, configuration=configuration, verbose=verbose)

    if verbose>=0:
        print(f"startFitness: {start_fitness}\nendFitness:   {end_fitness}")
    if log_data:
        # if data is being saved, save the end fiteness
        datalogger.data["end_fitness"] = end_fitness
        datalogger.data["init_flux"] = init_flux
        datalogger.data["generator_type"] = generator_type
        datalogger.data["neuron_configuration"] = configuration
        datalogger.data["start_fitness"] = start_fitness
        datalogger.data["generator_type"] = generator_type
        if "tolerance" in tracking_parameters:
            datalogger.data['tolerance'] = tolerance
        datafiles = os.listdir("./data")
        existing_files = [int(name.split("i")[-1].split(".")[0]) for name in datafiles if f"{generator_type}-{int(np.round(start_fitness, 5)*100000)}-{int(np.round(end_fitness, 5)*100000)}" in name]
        if not filename:
            if len(existing_files)>0:
                iteration = max(existing_files)+1
                filename = f"{generator_type}-{int(np.round(start_fitness,5)*100000)}-{int(np.round(end_fitness,5)*100000)}i{iteration}"
            else:
                filename = f"{generator_type}-{int(np.round(start_fitness,5)*100000)}-{int(np.round(end_fitness,5)*100000)}i0"
        if folderName:
            filepath = Path(f"./data/{folderName}/{filename}")
        else:
            filepath = Path(f"./data/{filename}")
        datalogger.save(filepath)
        print(filepath)
        return filename
    else:
        print("Not saved")
        return "not saved"
