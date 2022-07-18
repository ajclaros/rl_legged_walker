import numpy as np
import json


# Parent class which can be extended to provide additional functionality
# To help keep track of matrix dimensions I use
# N for the size of the network (fully connected neurons, including self loops NxN )

class CTRNN():

    #Constructor including boundaries of acceptable range
    def __init__(self, size, weight_range=16, bias_range=16, tc_min=1, tc_max=1, WR =16.0, BR = 16.0, TR = 5.0, TA=6.0):
        self.size = size                                # number of neurons in the network (N)
        self.voltages = np.zeros(size)                  # neuron activation vector
        self.time_constants = np.ones(size)             # time-constant vector
        self.biases = np.zeros(size)                    # bias vector
        self.weights_arr = np.zeros((size, size))     # inner weight matrix NxN
        self.outputs = np.zeros(size)                   # neuron output vector
        self.inputs = np.zeros(size)
        self.inv_time_constants = 1/self.time_constants
        self.WR = WR
        self.BR = BR
        self.TR = TR
        self.TA = TA




        #Parameter ranges - FIXED values
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.tc_min = tc_min
        self.tc_max = tc_max

    def reset(self):
        self.voltages = np.zeros(self.size)
        self.time_constants = np.ones(self.size)
        self.biases = np.zeros(self.size)
        self.inner_weights = np.zeros((self.size, self.size))
        self.outputs = np.zeros(self.size)

    # allow runs to be easily reproduced while still assigning random initial starting states
    def randomize_parameters_with_seed(self, seed):
        np.random.seed(seed)
        self.randomize_parameters()


    def setSize(self, size):
        self.size = size

    def setVoltages(self, voltages):
        self.voltages = voltages

    def setTimeConstants(self, time_constants):
        self.time_constants = time_constants * self.TR+self.TA
        self.inv_time_constants = 1.0/self.time_constants

    def setBiases(self, biases):
        self.biases = biases *self.BR

    def setWeights(self, inner_weights):
        self.inner_weights = inner_weights * self.WR

    def setInputs(self, inputs):
        self.inputs = inputs

    def randomize_parameters(self):
        self.inner_weights = np.random.uniform(-self.weight_range, self.weight_range,size=(self.size,self.size))
        self.biases = np.random.uniform(-self.bias_range, self.bias_range,size=(self.size))
        self.time_constants = np.random.uniform(self.tc_min, self.tc_max, size=(self.size))
        self.inv_time_constants = 1.0/self.time_constants

    def initializeState(self,v):
        self.voltages = v
        self.inv_time_constants = 1.0/self.time_constants
        self.outputs = sigmoid( self.voltages + self.biases)

    def recoverParameters(self):
        return np.append(self.inner_weights.reshape(self.inner_weights.size)/self.WR, [self.biases/self.BR, (self.time_constants-self.TA)/self.TR])

    #step without input - used for oscillator task
    def step(self,dt):
        netinput = self.inputs + np.dot(self.inner_weights.T, self.outputs)
        self.voltages += dt * (self.inv_time_constants * (-self.voltages + netinput))
        self.outputs = sigmoid(self.voltages+self.biases)

    def getBounds(self):
        return [self.size, self.weight_range, self.bias_range, self.tc_min, self.tc_max]

    def setBounds(self, size, weight_range, bias_range, tc_min, tc_max):
        self.size, self.weight_range, self.bias_range, self.tc_min, self.tc_max = size, weight_range, bias_range, tc_min, tc_max

    #always go through weights, bias, time constants
    def get_normalized_parameters(self):
        #                  NxN w           bias, tc
        genesize = self.size*self.size + 2*self.size
        genes = np.zeros(genesize)
        k=0
        for i in range( self.size):
            for j in range(self.size):
                genes[k] =  self.inner_weights[i][j] / self.weight_range
                k+=1
        for i in range(self.size):
            genes[k] = self.biases[i] / self.bias_range
            k+=1
        for i in range(self.size):
            if self.tc_max == self.tc_min:
                genes[k] = self.tc_max
            else:
                genes[k] = ((self.time_constants[i] - self.tc_min) / (self.tc_max-self.tc_min) * 2 - 1 )
            k+=1
        return genes

    #IMPORTANT: always go through weights, bias, time constants
    def set_normalized_parameters(self, genotype):
        k = 0
        for i in range(self.size):
            for j in range(self.size):
                self.inner_weights[i][j] = genotype[k]*self.weight_range
                k += 1
        for i in range(self.size):
            self.biases[i] = genotype[k]*self.bias_range
            k += 1
        for i in range(self.size):
            if self.tc_max == self.tc_min:
                self.time_constants[i] = self.tc_min
            else:
                self.time_constants[i] = ((genotype[k] + 1)/2)*(self.tc_max-self.tc_min) + self.tc_min
            k += 1
        self.inv_time_constants = 1.0/self.time_constants

    # Pretty print option to help debug
    def pprint( self ):
        for k, v in self.__dict__.items():
            print( f"'{k}' : {v}")

    #Goal here is to provide a human readable encoding of the CTRNN, likewise simple to load
    def save_json( self, filename="temp_ctrnn.json"):
        nn_json = json.dumps( self.__dict__, default=default,indent=4 )
        with open( filename, 'w') as filehandle:
            filehandle.write( nn_json )
        filehandle.close()

    def load_json( self, filename="temp_ctrnn.json"):
        with open( filename, 'r') as filehandle:
            nn = json.loads( filehandle.read() )
        for key,value in nn.items():
            #TODO fix this at some point, but for now assume np.array
            if isinstance(value, list):
                value = np.array(value)
            setattr(self, key, value)

# Used because np.array is not serializable for JSON
# There might be better options, but this works for now
# Approach from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def sigmoid(x):
    return 1/(1+np.exp(-x))
