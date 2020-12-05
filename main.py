import numpy as np
import tensorflow as tf

from models.net import NeuralNet
from models.utilities import custom_loss
from data.SyntheticMatrices import SyntheticMatrixSet

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
    n_epochs = 500
    n_samples = 1000
    dim = 5
    rank = 4
    sparsity = 0.5

    test_set_size = int(0.2 * n_samples)

    data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
    U, L, S, M = data_generator.generate_set(n_matrices=n_samples)

    # Split data set.
    U_test, L_test, S_test, M_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size]
    U_train, L_train, S_train, M_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:]

    net = NeuralNet(n_epochs=n_epochs, input_dim=dim, output_dim=(dim, rank), loss=custom_loss)
    net.train(X=M_train, y=M_train)
    net.plot_metric(metric='loss')

    score = net.evaluate(X_test=M_test, y_test=M_test)

