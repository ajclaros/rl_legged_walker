import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import sys

if len( sys.argv ) > 1:
    load = sys.argv[1]
else:
    load="jason.csv"

data = np.genfromtxt(load,delimiter=",", dtype=float, names=True)

print( shape(data[1]) )


w_a_label=data.dtype.names[0]
w_b_label=data.dtype.names[1]

alpha=1
length=len(data)
plot_data=np.zeros( (2,length ) )
colors=[]


for i in range(len(data)):
    if i %100==0:
        print(i)

    plot_data[0][i] = w_a = data[i][0]
    plot_data[1][i] = w_a = data[i][1]
    fit=data[i][2]
    if fit > 1.0:
        fit = 1.0
    
    colors.append( [fit, fit, fit, alpha] )
    
    # w_b = data[i][1]
    # fit = data[i][2]

wAs=plot_data[0]
wBs=plot_data[1]



plt.scatter(wAs, wBs, c=colors )
plt.xlabel(w_a_label)
plt.ylabel(w_b_label)

plt.show()