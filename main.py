import numpy as np
import tensorflow as tf

from models.net import NeuralNet
from models.utilities import custom_loss
from utilities.metrics import MatrixSparsity, MatrixRank
from utilities.plot import plot_matrices
from data.SyntheticMatrices import SyntheticMatrixSet

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    n_epochs = 2
    n_samples = 1000
    dim = 20
    rank = 3
    sparsity = 0.95

    test_set_size = int(0.2 * n_samples)

    print(f'starting generating {n_samples} {dim}x{dim} matrices with rank {rank} and sparsity {sparsity}...')
    data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
    U, L, S, M, M_tri = data_generator.generate_set(n_matrices=n_samples)

    # Split data set.
    U_test, L_test, S_test, M_test, M_tri_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    U_train, L_train, S_train, M_train, M_tri_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]

    tri_dim = M_tri_train.shape[1]
    net = NeuralNet(n_epochs=n_epochs, input_dim=tri_dim, output_dim=(dim, rank),
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim), MatrixRank(dim=dim)])
    print(f'starting training for {n_epochs} epochs on {M_train.shape[0]} matrices...')
    net.train(X=M_tri_train, y=M_train)
    net.plot_metrics(metrics=['loss', 'sparsity', 'rank'])

    print(f'starting evaluating of the network on {M_test.shape[0]} matrices...')
    # Evaluate
    U_pred = net.predict(X=M_tri_test)

    matrix_sparsity = MatrixSparsity(dim=dim)
    matrix_sparsity.update_state(M_true=M_test, U_pred=U_pred)
    sparsity_test = matrix_sparsity.result()
    tf.print(f'Sparsity: {sparsity_test}')

    matrix_rank = MatrixRank(dim=dim)
    matrix_rank.update_state(M_true=M_test, U_pred=U_pred)
    rank_test = matrix_rank.result()
    tf.print(f'Sparsity: {rank_test}')

    # Plot one example
    fig = plot_matrices(M_test[0], U_pred[0])
    fig.show()


if __name__ == '__main__':
    main()
