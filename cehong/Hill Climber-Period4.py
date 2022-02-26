#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from ctrnn import CTRNN
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from Utilities import *


# In[3]:


nn = CTRNN(2, weight_range=weight_range, bias_range=bias_range, tc_min=tc_min, tc_max=tc_max)
n_simulation = 20

target_period = 4
target_freq = 1/target_period

freq_ratios = np.linspace(0.1,0.9,9).round(3)
n_step = 60

results = pd.DataFrame(columns = ['jump_size','Ratio','clip_change_max','Fitness','Period_performance','Change_performance'])
nn.reset()
index = 0
np.random.seed(1)

for jump_size in np.linspace(0.1,0.4,4).round(1):
    print("------------------------Finished", (jump_size-0.1)*250, '%-----------------------------------')
    for clipmax in [0.25,0.5,0.75,1]:
        for freq_ratio in freq_ratios:

            fitnesses = np.zeros(n_simulation)
            freq_perfs = np.zeros(n_simulation)
            change_perfs = np.zeros(n_simulation)
            for k in range(n_simulation):
                old = 0
                param_summary = []
                nn.reset()
                other = np.array([5.15 , -10.75,16,16 ])/16
                nn.initializeState(np.array([0.0,0.0]))
                weights = np.array([4.9,16 ,- 16 , 4.71])/16
                best = 0
                best_weight = 0


                freqs = np.zeros(n_step)
                best_fit = np.zeros(n_step)



                for i in range(n_step):
                    nn.reset()
                    weights_new = weights + (np.random.rand(4)*2-1)*jump_size
                    weights_new = np.clip(weights_new,-16,16)
                    params_new = np.append(weights_new,other)

                    nn.set_normalized_parameters(params_new)


                    period, change = frequency_fitness_no_transients(nn, combo_func=None, show_plots = False, init_duration=10, test_duration=40,                     clip_change_max=clipmax, min_freq_score=0.001, stepsize=0.01, target_period=target_period)
                    fitness = freq_ratio * period + (1-freq_ratio) * change

                    if fitness > old:
                        old = fitness 
                        weights = weights_new

                nn.reset()
                nn.set_normalized_parameters(np.append(weights,other))
                period, change = frequency_fitness_no_transients(nn, combo_func=None, show_plots = False, init_duration=10, test_duration=30,                     clip_change_max=1, min_freq_score=0.001, stepsize=0.01, target_period=target_period)
                fitness = 0.5* period + 0.5 * change
                print("The fitness is:",fitness)
                results.loc[index,:] = [jump_size,freq_ratio,clipmax,fitness,period,change]
                index += 1
        
print("Finished")
results = results.astype('float64')
results.to_csv('./result_of_hc_full_period4_step_60.csv')


# In[4]:


gp = results.groupby(by = ['Ratio','clip_change_max','jump_size'])
print(gp.mean().sort_values('Fitness',ascending=False).head(10))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




