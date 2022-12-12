import leggedwalker
import numpy as np
from ctrnn import CTRNN
from datalogger import DataLogger

# Task Parameters
duration = 220.0
stepsize = 0.1
# time = np.arange(0.0, duration, stepsize)


def fitnessFunction(
    genotype,
    duration=220.0,
    N=2,
    generator_type="RPG",
    configuration=[0],
    verbose=0,
    record=False,
    stepsize=0.1,
    filename=None,
):
    # Create the agent's body
    legged = leggedwalker.LeggedAgent()
    # Create the nervous system
    ns = CTRNN(N)
    # Set the parameters of the nervous system according to the genotype-phenotype map
    weights = genotype[0 : N * N]
    ns.setWeights(weights.reshape((N, N)))
    ns.setBiases(genotype[N * N : N * N + N])
    ns.setTimeConstants(genotype[N * N + N :])
    # Initialize the state of the nervous system to some value
    ns.initializeState(np.zeros(N))
    # learner = RL_CTRNN(ns)
    # Loop through simulated time, use Euler Method to step the nervous system and body
    time = np.arange(0.0, duration, stepsize)
    if record:
        datalogger = DataLogger()
        datalogger.data["sample_rate"] = 1
        datalogger.data["stepsize"] = stepsize
        datalogger.data["learning_start"] = 0
        datalogger.data["duration"] = duration
        datalogger.data["size"] = N
        datalogger.data["outputs"] = np.zeros(shape=(len(time), N))
        datalogger.data["distance"] = np.zeros(shape=(len(time)))
        datalogger.data["omega"] = np.zeros(shape=(len(time)))
        datalogger.data["angle"] = np.zeros(shape=(len(time)))
        datalogger.data["footstate"] = np.zeros(shape=(len(time)))
        datalogger.data["footX"] = np.zeros(shape=(len(time)))
        datalogger.data["footY"] = np.zeros(shape=(len(time)))
        datalogger.data["jointX"] = np.zeros(shape=(len(time)))
        datalogger.data["jointY"] = np.zeros(shape=(len(time)))
        datalogger.data["vx"] = np.zeros(shape=(len(time)))
    for i, t in enumerate(time):
        if record:
            datalogger.data["outputs"][i] = ns.outputs
            datalogger.data["distance"][i] = legged.cx
            datalogger.data["omega"][i] = legged.omega
            datalogger.data["angle"][i] = legged.angle
            datalogger.data["footX"][i] = legged.footX
            datalogger.data["footY"][i] = legged.footY
            datalogger.data["vx"][i] = legged.vx
            datalogger.data["footstate"][i] = legged.footstate
            datalogger.data["jointX"][i] = legged.jointX
            datalogger.data["jointY"][i] = legged.jointY
        if generator_type == "RPG":
            ns.setInputs(
                np.array([legged.anglefeedback()] * N)
            )  # Set neuron input to angle feedback based on current body state
        else:
            ns.setInputs(np.array([0] * ns.size))
        #        ns.setInputs(np.array([0.0]*N))  # Set neuron input to angle feedback based on current body state
        ns.step(stepsize)  # Update the nervous system based on inputs
        # legged.step3(stepsize, ns.outputs)
        legged.stepN(stepsize, ns.outputs, configuration)
        # Update the body based on nervous system activity
    #        fitness_arr[i] = body.cx                        # track position of body
    # update neurons based on speed of movement (cx(t)-cx(t-1))/dt
    # Calculate the fitness based on distance covered over the duration of time
    fit = legged.cx / duration
    if record:
        datalogger.data["start_fitness"] = fit
        datalogger.data["end_fitness"] = fit
        if filename:
            datalogger.save(f"./data/runs/{filename}")
        else:
            filename = f"behavior-{generator_type}-{int(np.round(fit,5)*100000)}-s{ns.size}-c{'_'.join(str(num) for num in configuration)}"
            datalogger.save(f"./data/runs/{filename}")
    return fit
