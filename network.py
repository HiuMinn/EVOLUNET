import numpy as np

class EVOLUNET():
    def __init__(self, size):
        self.size = size
        self.num_layers = len(self.size)
        self.bias = 0
    