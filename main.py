import numpy as np
import tensorflow as tf

from models.net import NeuralNet
from models.utilities import custom_loss
from utilities.metrics import MatrixSparsity, MatrixRank
from data.SyntheticMatrices import SyntheticMatrixSet

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


if __name__ == '__main__':
    n_epochs = 300
    n_samples = 100000
    dim = 20
    rank = 3
    sparsity = 0.95

    test_set_size = int(0.2 * n_samples)

    data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
    U, L, S, M, M_tri = data_generator.generate_set(n_matrices=n_samples)

    # Split data set.
    U_test, L_test, S_test, M_test, M_tri_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    U_train, L_train, S_train, M_train, M_tri_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]

    tri_dim = M_tri_train.shape[1]
    net = NeuralNet(n_epochs=n_epochs, input_dim=tri_dim, output_dim=(dim, rank),
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim), MatrixRank(dim=dim)])
    net.train(X=M_tri_train, y=M_train)
    net.plot_metrics(metrics=['loss', 'sparsity', 'rank'])

    score = net.evaluate(X_test=M_tri_test, y_test=M_test)
