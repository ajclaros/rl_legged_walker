import numpy as np


class DataLogger():

    def __init__(self):
        self.data = {}

    def save(self, filename):
        kwargs = {key: self.data[key] for key in list(self.data.keys())}
        np.savez(filename, **kwargs)
