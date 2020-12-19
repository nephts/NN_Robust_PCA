import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models.net import NeuralNet
from models.utilities import custom_loss
from utilities.metrics import MatrixSparsity
from utilities.plot import plot_matrices
from data.SyntheticMatrices import SyntheticMatrixSet
from pcp import pcp

from psychdata import psych

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_data(data='synthetic', dim=None, rank=None, sparsity=None, n_samples=None):
    if data == 'synthetic' and dim is None or rank is None or sparsity is None or n_samples is None:
        raise ValueError('If you want synthetic data, you have to specify dim, rank, sparsity and n_samples')

    if data == 'synthetic':
        print(f'starting generating {n_samples} {dim}x{dim} matrices with rank {rank} and sparsity {sparsity}...')
        data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
        U, L, S, M, M_tri = data_generator.generate_set(n_matrices=n_samples)
        return U, L, S, M, M_tri


def main():
    data = 'synthetic'
    n_epochs = 300
    n_samples = 100000
    dim = 27
    rank = 5
    sparsity = 0.95

    test_set_size = int(0.2 * n_samples)

    U, L, S, M, M_tri = get_data(data=data, dim=dim, rank=rank, sparsity=sparsity, n_samples=n_samples)
    # Split data set.
    U_test, L_test, S_test, M_test, M_tri_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    U_train, L_train, S_train, M_train, M_tri_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]

    net = NeuralNet(n_epochs=n_epochs, dim=dim, output_dim=(dim, rank),
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim)])

    # Train
    print(f'starting training for {n_epochs} epochs on {M_train.shape[0]} matrices...')
    net.train(X=M_tri_train, y=M_train)
    net.plot_metrics(metrics=['loss', 'sparsity'])

    # Evaluate on psychdata
    psychdata = psych.Psychdata()
    data = psychdata.get_corr()
    # DENISE
    print(f'starting evaluating of the network on {M_test.shape[0]} matrices...')
    score, U_denise = net.evaluate(M_test=data, dim=dim)
    L_denise = np.matmul(U_denise[0], U_denise[0].T)
    S_denise = data - L_denise
    tf.print(f'Sparsity (DENISE): {score}')

    # PCP
    L_pcp, S_pcp, (u_pcp, s_pcp, v_pcp) = pcp(M=data)
    matrix_sparsity = MatrixSparsity(dim=dim)
    matrix_sparsity.update_state(M_true=M_test, L_pred=L_pcp)
    score_pcp = matrix_sparsity.result()
    tf.print(f'Sparsity (PCP): {score_pcp}')

    with open("plots/res.txt", "w") as output:
        output.write(f"DENISE trained {n_epochs} epochs with {n_samples - test_set_size} {dim}x{dim} rank {rank} "
                     f"matrices.\nDENISE: \n\nL: \n {str(L_denise)}\n\nS: \n{str(S_denise)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\nPCP: \n\nL: \n{str(L_pcp)}\n\nS: \n{str(S_pcp)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\nM: \n{str(data)}")

    fig = plot_matrices(data, L_pred=L_pcp)
    plt.savefig('plots/pcp_output.pdf')
    fig.show()


if __name__ == '__main__':
    main()
