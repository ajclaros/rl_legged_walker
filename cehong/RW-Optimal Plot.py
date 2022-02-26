#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
from ctrnn import CTRNN
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from Utilities import *


# In[2]:


nn = CTRNN(2, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max)
n_simulation = 10
other = np.array([5.15 , -10.75,16,16 ])/16
clipmax = 1
freq_ratio = 0.4
jump_size = 0.25

target_period = 4
target_freq = 1/target_period
    

n_step = 150

results = pd.DataFrame(columns = ['jump_size','Ratio','clip_change_max','Fitness','Period_performance','Change_performance'])
nn.reset()
index = 0
np.random.seed(0)

fitness_summary = pd.DataFrame()


for k in range(n_simulation):

    old = 0
    nn.initializeState(np.array([0.0,0.0]))
    weights = np.array([4.9,16 ,- 16 , 4.71])/16
    best_fitnesses = []
    fitnesses = []

    for i in range(n_step):
        nn.reset()
        weights_new = weights + (np.random.rand(4)*2-1)*jump_size
        weights_new = np.clip(weights_new,-16,16)
        params_new = np.append(weights_new,other)

        nn.set_normalized_parameters(params_new)


        period, change = frequency_fitness_no_transients(nn, combo_func=None, show_plots = False, init_duration=10, test_duration=40,         clip_change_max=clipmax, min_freq_score=0.001, stepsize=0.01, target_period=target_period)
        fitness = freq_ratio * period + (1-freq_ratio) * change
        weights = weights_new
        if fitness > old:
            old = fitness 
            best_weights = weights_new
        fitnesses.append(fitness)
        best_fitnesses.append(old)

    nn.reset()
    nn.set_normalized_parameters(np.append(best_weights,other))
    period, change = frequency_fitness_no_transients(nn, combo_func=None, show_plots = False, init_duration=10, test_duration=30,         clip_change_max=1, min_freq_score=0.001, stepsize=0.01, target_period=target_period)
    fitness = 0.5* period + 0.5 * change
    print("The fitness is:",fitness)
#     results.loc[index,:] = [jump_size,freq_ratio,clipmax,fitness,period,change]
    fitness_summary['Fitness'] = fitnesses
    fitness_summary['Best fitness'] = best_fitnesses

    path = './Data/fitnesses_period4_RW_simulation' + str(k) + '.csv'
    fitness_summary.to_csv(path)


# In[3]:


fitness_summary.reset_index().plot.line('index' , ['Fitness','Best fitness'])
plt.show()


# In[ ]:





# In[ ]:




