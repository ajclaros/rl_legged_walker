import numpy as np
import matplotlib.pyplot as plt
import random
import glob
import math

import statistics 
import csv


#work-in-progress needs to be adapted



# #assuming all files in this folder are the same
# #experiment only differing in seeds
# files = glob.glob("CSV2/*")


# #will set this once we have read one file
# generations = 0

# #this will be a list of lists (the inner lists to be all the values of a given generation)
# all_fitness_by_generation = []

# #sample size is the number of data points
# #separate data files in this case
# n=len(files)

# #read one file to setup lists
# with open(files[0], newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:        
#         generations+=1
#         #add a list to 
#         all_fitness_by_generation.append( [ int(row["min_fitness"] )  ] )

# #skip first file
# for file in files[1:]:
    
#     with open(file, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         g=0
#         for row in reader:            
#             #adding the int() to force python to treat as numbers
#             #otherwise it treats as strings and y axis will be out of order
#             #this will obviously need to be replaced by what you use in your files
#             all_fitness_by_generation[g].append( int(row["min_fitness"] )  )
#             g+=1


# #lists to keep track of the mean, lower/upper bound of error relative to mean
# mean_values=[]
# lower_errors=[]
# upper_errors=[]

# for g in range( generations ):
#     mean = statistics.mean( all_fitness_by_generation[g])
#     #mean calculated and stored for plotting later
#     mean_values.append(  mean )
#     #calculate the standard deviation
#     sd = statistics.stdev(all_fitness_by_generation[g])
#     lower_errors.append( mean - sd )
#     upper_errors.append( mean + sd )

# #simple produce a list with 0....to (generations-1) in it
# x = range(generations)
# y = mean_values

# #draw mean line
# plt.plot(x, y, color="red", label="fitness" )
# plt.yscale("log")


# #draw shaded region from lower to upper
# plt.fill_between(x, lower_errors, upper_errors, alpha=0.25, facecolor="red", label="standard deviation")

# #show a legend
# plt.legend()

# #add labels to the axes
# plt.xlabel('generation')
# plt.ylabel('fitness')

# #saves to file
# plt.savefig('agg_demo_plot.png')

# #shows on the screen
# plt.show()



