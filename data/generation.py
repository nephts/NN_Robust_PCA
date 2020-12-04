import numpy as np


class DataGenerator:
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim

    def generate(self):
        X = np.random.rand(self.size, self.dim, self.dim)
        y = np.random.rand(self.size, 5, 5)
        return X, y
