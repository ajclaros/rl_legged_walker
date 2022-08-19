import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import leggedwalker
from rl_ctrnn import RL_CTRNN
from walking_task import WalkingTask
import numpy as np

style.use('fivethirtyeight')
fig = plt.figure()
#ax1 = fig.add_subpllot(1,1,1)
def animate(i):
    pass


params = {
    "window_size": 400,             #unit seconds
    "learn_rate": 0.008,
    "conv_rate": 0.004,
    "min_period": 300,              #unit seconds
    "max_period": 800,              #unit seconds
    "init_flux": 6,
    "max_flux": 8,
    "duration": 1000,               #unit seconds
    "size": 2,
#    "generator_type": "RPG",
#    "tolerance": 0.00,
#    "neuron_configuration": [0]
}

params['bias_init_flux'] = params['init_flux']
params['init_flux'] = params['init_flux']
params['max_flux'] = params['max_flux']
params['bias_init_flux'] = params['init_flux']
params['bias_init_flux'] = params['init_flux']
params['bias_max_flux'] = params['max_flux']
params['bias_max_flux'] = params['max_flux']
params['bias_min_period'] = params['min_period']
params['bias_max_period'] = params['max_period']
params['bias_conv_rate'] = params['conv_rate']

learner = WalkingTask(**params)
starting_genome=np.array([0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193])
size = 2
weights = starting_genome[0 : size* size]
learner.setWeights(weights.reshape((size, size)))
learner.setBiases(starting_genome[size* size: size* size+ size])
learner.setTimeConstants(starting_genome[size* size+size:])
learner.initializeState(np.zeros(size))
body = leggedwalker.LeggedAgent()
learner.simulate(body, learning_start=params['window_size'], verbose=.1, generator_type='RPG', neuron_configuration=[0])
