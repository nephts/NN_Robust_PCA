import numpy as np
import tensorflow as tf

from models.net import NeuralNet
from models.utilities import custom_loss
from data.generation import DataGenerator

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    n_samples = 100
    dim = 10

    data_generator = DataGenerator(size=n_samples, dim=dim)
    X, y = data_generator.generate()

    net = NeuralNet(input_dim=dim, output_dim=5, loss=custom_loss)
    net.train(X=X, y=y)
    net.plot_metric(metric='loss')
