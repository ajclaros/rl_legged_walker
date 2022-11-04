import numpy as np
import leggedwalker
from ctrnn import CTRNN
import matplotlib.pyplot as plt
import os

folderName = "RPG-s3-01-d30000"
files = os.listdir(f"./data/{folderName}")
files = [name for name in files if "npz" in name]

data = np.load(f"./data/{folderName}/{files[0]}")
thresh = 0.015
add = 0.001
indices = np.where((abs(data['performance_average_hist'])>thresh) & (abs(data['performance_average_hist'])<thresh+add))[0]
diff = indices -np.roll(indices, 1)
indices = indices[np.where(abs(diff)>1)]
boundary = 45

startix = 4
#plt.plot(data['distance'])
#plt.plot(data['performance_average_hist'][indices[startix]-boundary:indices[startix+1]+boundary])
#plt.plot(data['extended_weights'].T[0][0][indices[startix]-boundary:indices[startix+1]+boundary])
#plt.show()

N = 3
stepsize = 0.1
generator_type = "RPG"
ns = CTRNN(N)
ns.inner_weights = data['extended_weights'][:,:][indices[startix]-boundary]
ns.biases = data['biases'][:][indices[startix]-boundary]
ns.setTimeConstants(data['startgenome'][3*3+3:])
ns.initializeState(N)
legged = leggedwalker.LeggedAgent()
time = np.arange(0.0, 220, 0.1)
mult = 1
# every weight will run
# for i in range(int((indices[startix+1]+boundary-(indices[startix]-boundary))/mult)):
#     for j, t in enumerate(time):
#         if generator_type == "RPG":
#             ns.setInputs(
#                 np.array([legged.anglefeedback()]*N)
#             )
#         else:
#             ns.setInputs(np.array([0]*N))
#         ns.step(stepsize)
#         legged.stepN(stepsize, ns.outputs, [0,1])
#     print(legged.cx/220, i)
#     legged = leggedwalker.LeggedAgent()
#     ns.inner_weights = data['extended_weights'][:,:][indices[startix]-boundary+i*mult]
#     ns.biases = data['biases'][:][indices[startix]-boundary+i*mult]
#     ns.reset()
# plt.show()



trial= {}
trial['distance'] = np.zeros(indices[startix+1]+boundary -(indices[startix]-boundary))
for i in range(int((indices[startix+1]+boundary-(indices[startix]-boundary)))):
    print(i, flush=False, sep=' ')
    if generator_type == "RPG":
        ns.setInputs(
            np.array([legged.anglefeedback()]*N)
        )
    else:
        ns.setInputs(np.array([0]*N))
    ns.step(stepsize)
    legged.stepN(stepsize, ns.outputs, [0,1])
    ns.inner_weights = data['extended_weights'][:,:][indices[startix]-boundary+i]
    ns.biases = data['biases'][:][indices[startix]-boundary+i]
    trial['distance'][i]= legged.cx
