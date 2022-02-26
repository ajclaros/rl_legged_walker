import matplotlib.pyplot as plt

import numpy as np


data = np.genfromtxt("jason/data/figure2.dat",delimiter=",", dtype=float)

print(data[0])
threshold=0.4
success=0
for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=min(1, max(0, final_fit/init_fit/4))
    start=0
    if final_fit>threshold:
        alpha=1
        success+=1
    else:
        alpha=0.1
    plt.plot( range(500-start), data[index][start+5:], color=[improvement,0,0,final_fit] )
plt.title(f"success: {100*success/len(data):0.3f}%")
plt.show()



for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=max(0,final_fit-init_fit)

    plt.scatter( init_fit, final_fit, color=[improvement,0,0,0.25] )
plt.plot( [0,1], [0,1], color=[0,0,0,0.25] )
plt.title(f"success: {100*success/len(data):0.3f}%")
plt.xlabel("init_fit")
plt.ylabel("final_fit")
plt.xlim(0,.8)
plt.ylim(0,.8)

plt.show()

for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=max(0,final_fit-init_fit)


    plt.scatter( init_dist, final_dist, color=[final_fit,0,0,0.25] )
plt.title(f"success: {100*success/len(data):0.3f}%")
plt.xlabel("init_dist")
plt.ylabel("final_dist")
    

plt.show()

for index in range(len(data)):
    #init_fit,final_fit,init_est_dist,final_est_dist
    init_fit=data[index][0]
    final_fit=data[index][1]
    init_dist=data[index][2]
    final_dist=data[index][3]
    improvement=max(0,final_fit-init_fit)

    plt.scatter( init_dist, final_fit, color=[improvement,0,0,0.25] )
    plt.xlabel("init_dist")
    plt.ylabel("final_fit")
    

plt.show()

   
