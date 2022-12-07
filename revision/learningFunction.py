import leggedwalker
import numpy as np
from rl_ctrnn import RL_CTRNN
from ctrnn import CTRNN
from walking_task import WalkingTask
import datetime
from datalogger import DataLogger
from fitnessFunction import fitnessFunction
from pathlib import Path
from csv import DictWriter
import time
import os


def append_dict_as_row(file_name, field_names, elements):
    # Open file in append mode
    with open(file_name, "a+", newline="\n") as write_obj:
        # Create a writer object from csv module
        csv_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(elements)


def learn(
    starting_genome,
    duration=2000,
    size=2,
    window_size=4000,
    stepsize=0.1,
    init_flux=2.75,
    max_flux=10,
    min_period=300,
    max_period=400,
    conv_rate=0.004,
    learn_rate=0.004,
    bias_init_flux=2.75,
    bias_max_flux=10,
    bias_min_period=300,
    bias_max_period=400,
    bias_conv_rate=0.004,
    log_data=False,
    verbose=1.00,
    generator_type="RPG",
    neuron_configuration=[0],
    tracking_parameters=None,
    filename=None,
    folderName=None,
    print_done=False,
    trial=None,
    csv_name=None,
    genome_num=None,
    performance_func=None,
    reward_func=None,
    record_every=1,
    learning_start=0,
    starting_fitness=None,
):

    np.random.seed()
    learner = WalkingTask(
        duration=duration,
        size=size,
        stepsize=stepsize,
        running_window_mode=True,
        running_window_size=window_size,
        init_flux_amp=init_flux,
        max_flux_amp=max_flux,
        flux_period_min=min_period,
        flux_period_max=max_period,
        flux_conv_rate=conv_rate,
        learn_rate=learn_rate,
        bias_init_flux_amp=bias_init_flux,
        bias_max_flux_amp=bias_max_flux,
        bias_flux_period_min=min_period,
        bias_flux_period_max=max_period,
        bias_flux_conv_rate=conv_rate,
        performance_func=performance_func,
        reward_func=reward_func,
        record_every=record_every,
    )
    weights = starting_genome[0 : size * size]
    learner.setWeights(weights.reshape((size, size)))
    learner.setBiases(starting_genome[size * size : size * size + size])
    learner.setTimeConstants(starting_genome[size * size + size :])
    learner.initializeState(np.zeros(size))
    body = leggedwalker.LeggedAgent()
    if log_data:
        datalogger = DataLogger()
        for var in tracking_parameters:
            if "weight" in var:
                datalogger.data[var] = np.zeros(
                    (int(duration / stepsize / record_every), size, size)
                )

            elif "bias" in var or "voltage" in var or "outputs" in var:
                datalogger.data[var] = np.zeros(
                    (int(duration / stepsize / record_every), size)
                )
            else:
                datalogger.data[var] = np.zeros(int(duration / stepsize / record_every))
        learner.simulate(
            body,
            learning_start=learning_start,
            datalogger=datalogger,
            verbose=verbose,
            generator_type=generator_type,
            configuration=neuron_configuration,
        )
    else:
        learner.simulate(
            body,
            learning_start=learning_start,
            generator_type=generator_type,
            configuration=neuron_configuration,
            verbose=verbose,
        )

    if starting_fitness == None:
        starting_fitness = fitnessFunction(
            starting_genome,
            N=size,
            generator_type=generator_type,
            configuration=neuron_configuration,
        )
    if reward_func == None:
        params = learner.recoverParameters(inner=False)
    else:
        params = learner.recoverParameters(inner=True)
    end_fitness = fitnessFunction(
        params,
        N=size,
        generator_type=generator_type,
        configuration=neuron_configuration,
        verbose=verbose,
    )
    end_performance = learner.performance_track.mean()

    # if verbose>=0:
    #    print(f"startFitness: {start_fitness}\nendFitness:   {end_fitness}")

    if csv_name:
        configuration = [str(a) for a in neuron_configuration]
        configuration = "|".join(configuration)
        configuration = "|" + configuration + "|"
        csv = os.listdir(f"./data/csv_focused/")
        csv = [name for name in csv if csv_name in name]  # [0]
        elements = {
            "start_fit": starting_fitness,
            "end_fit": end_fitness,
            "generator": generator_type,
            "configuration": configuration,
            "init_flux": init_flux,
            "max_flux": max_flux,
            "min_period": min_period,
            "max_period": max_period,
            "window_size": window_size,
            "genome_num": genome_num,
            "end_perf": end_performance,
            "rates": learn_rate,
        }
        append_dict_as_row(
            f"./data/csv_focused/{csv_name}",
            field_names=elements.keys(),
            elements=elements,
        )

    if log_data:
        # if data is being saved, save the end fiteness
        datalogger.data["end_fitness"] = end_fitness
        datalogger.data["init_flux"] = init_flux
        datalogger.data["generator_type"] = generator_type
        datalogger.data["neuron_configuration"] = neuron_configuration
        datalogger.data["start_fitness"] = starting_fitness
        datafiles = os.listdir(f"./data/{folderName}")
        existing_files = [
            int(name.split("i")[-1].split(".")[0])
            for name in datafiles
            if f"{generator_type}-{int(np.round(starting_fitness, 5)*100000)}-{int(np.round(end_fitness, 5)*100000)}"
            in name
        ]
        if not filename:
            if len(existing_files) > 0:
                iteration = max(existing_files) + 1
                filename = f"{generator_type}-{int(np.round(starting_fitness,5)*100000)}-{int(np.round(end_fitness,5)*100000)}i{iteration}"
            else:
                iteration = len(existing_files)
                filename = f"{generator_type}-{int(np.round(starting_fitness,5)*100000)}-{int(np.round(end_fitness,5)*100000)}-i{iteration}"
        if folderName:
            filepath = Path(f"./data/{folderName}/{filename}")
        else:
            filepath = Path(f"./data/{filename}")
        datalogger.save(filepath)

        if print_done:
            f"{trial} done"

        return filename
    else:
        if print_done:
            print(
                f"{generator_type}-{np.round(starting_fitness,5)}-{np.round(end_fitness,5)}"
            )
        return starting_fitness, end_fitness
