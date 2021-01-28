import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import time
import os
import pickle

from models.net import NeuralNet, NeuralNet_SVD
from models.utilities import custom_loss
from utilities.metrics import MatrixSparsity
from utilities.plot import plot_matrices, test_UV
from data.SyntheticMatrices import SyntheticMatrixSet, SyntheticMatrixSet_SVD
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

    if data == 'synthetic_SVD':
        print('dim needs to be n x m tuple')
        n = dim[0]
        m = dim[1]
        print(f'starting generating {n_samples} {n}x{m} matrices with rank {rank} and sparsity {sparsity}...')
        data_generator = SyntheticMatrixSet_SVD(dim=dim, rank=rank, sparsity=sparsity)
        U, V, L, S, M = data_generator.generate_set(n_matrices=n_samples)
        return U, V, L, S, M


def comparison(denise, method, data, dim, M_test, n_epochs, n_samples, test_set_size, rank):
    # DENISE
    score, U_denise = denise.evaluate(M_test=data, dim=dim)
    L_denise = np.matmul(U_denise[0], U_denise[0].T)
    S_denise = data - L_denise
    tf.print(f'Sparsity (DENISE): {score}')

    if method == 'pcp':
        L_method, S_method, (u_method, s_method, v_method) = pcp(M=data)
    matrix_sparsity = MatrixSparsity(dim=dim)
    matrix_sparsity.update_state(M_true=M_test, L_pred=L_method)
    score_pcp = matrix_sparsity.result()
    tf.print(f'Sparsity (PCP): {score_pcp}')

    with open("plots/res.txt", "w") as output:
        output.write(f"DENISE trained {n_epochs} epochs with {n_samples - test_set_size} {dim}x{dim} rank {rank} "
                     f"matrices.\nDENISE: \n\nL: \n {str(L_denise)}\n\nS: \n{str(S_denise)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\n{method}: \n\nL: \n{str(L_method)}\n\nS: \n{str(S_method)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\nM: \n{str(data)}\n\n")
        output.write(f"Relative Error of L : {tf.divide(tf.norm(L_denise - L_method, ord='fro', axis=(0, 1)), tf.norm(L_method, ord='fro', axis=(0, 1)))}\n")
        output.write(f"Relative Error of S: {tf.divide(tf.norm(S_denise - S_method, ord='fro', axis=(0, 1)), tf.norm(S_method, ord='fro', axis=(0, 1)))}\n")

    fig = plot_matrices(data, L_pred=L_method)
    plt.savefig(f'plots/{method}_output.pdf')
    fig.show()





def main():    
    # Load data --------------------------------------------------------------
    path = os.path.dirname(os.path.abspath(__file__))
    n_samples = 1000000
    dim = 10
    rank = 3
    nk = int(n_samples/1000)
    M = pickle.load( open( path + '/data/synthetic_matrices/M_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )
    M_tri = pickle.load( open( path + '/data/synthetic_matrices/M_tri_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )
    S = pickle.load( open( path + '/data/synthetic_matrices/S_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )

    # Split data set ---------------------------------------------------------
    test_set_size = int(0.2 * n_samples)
    S_test, M_test, M_tri_test = S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    S_train, M_train, M_tri_train = S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]
    
    # # Generate data ----------------------------------------------------------
    # data = 'synthetic'
    # n_samples = 10000
    # dim = 5
    # rank = 2
    # sparsity = 0.95
    # U, L, S, M, M_tri = get_data(data=data, dim=dim, rank=rank, sparsity=sparsity, n_samples=n_samples)
    
    # # Split data set ---------------------------------------------------------
    # test_set_size = int(0.2 * n_samples)
    # U_test, L_test, S_test, M_test, M_tri_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    # U_train, L_train, S_train, M_train, M_tri_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]

    # Network
    n_epochs = 2
    net = NeuralNet(n_epochs=n_epochs, dim=dim, output_dim=(dim, rank), batch_size=64,
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim)])

    # Train
    print(f'starting training for {n_epochs} epochs on {M_train.shape[0]} matrices...')
    net.train(X=M_tri_train, y=M_train)
    net.plot_metrics(metrics=['loss', 'sparsity'])

    # Compare
    # ''' !!! UNCOMMENT the following code for compare on psych data !!! '''
    # Compare on psychdata
    # psychdata = psych.Psychdata()
    # data = psychdata.get_corr()
    # comparison(denise=net, method='pcp', data=data, dim=dim, M_test=M_test, n_epochs=n_epochs,
    #          n_samples=n_samples, test_set_size=test_set_size, rank=rank)

    # ''' !!! UNCOMMENT the following code for compare on finance data !!! '''
    # # Compare on Finance data
    # with open('Stock prices dax 30.csv') as stockprices:
    #     data = list(csv.reader(stockprices, delimiter=";"))
    # data_array = np.array(data)
    # data_only = data_array[1:data_array.shape[0], 1:6].T
    # data_only = np.array([[float(y) for y in x] for x in data_only])
    # Sigma = np.zeros((5, 5))

    # for k in range(0, data_only.shape[1] - 1):
    #     Sigma = Sigma + np.dot(np.subtract(data_only[:, k], np.mean(data_only, axis=1)).reshape((5, 1)),
    #                             np.subtract(data_only[:, k], np.mean(data_only, axis=1)).reshape((1, 5)))

    # Sigma = (1 / (data_only.shape[1] - 1)) * Sigma

    # comparison(denise=net, method='pcp', data=Sigma, dim=dim, M_test=M_test, n_epochs=n_epochs,
    #             n_samples=n_samples, test_set_size=test_set_size, rank=rank)


def main_SVD():
    data = 'synthetic_SVD'
    n_epochs = 5
    iterations = 2
    n_samples = 500
    dim = (5,6)
    rank = 2
    sparsity = 0.95
    
    test_set_size = int(0.2 * n_samples)
    
    U, V, L, S, M = get_data(data=data, dim=dim, rank=rank, sparsity=sparsity, n_samples=n_samples)
    # Split data set.
    U_test, V_test, L_test, S_test, M_test = U[:test_set_size], V[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size]
    U_train, V_train, L_train, S_train, M_train = U[test_set_size:], V[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:]
    
    net = NeuralNet_SVD(rank=rank, n_epochs=n_epochs, iterations=iterations, dim=dim, batch_size=64,
                    loss=custom_loss)
    
    # # Train
    print(f'starting training for {n_epochs} epochs on {M_train.shape[0]} matrices...')
    net.train(X=M_train, y=M_train)
    net.plot_metrics(metrics=['loss'])
    
    test_UV(U_train, V_train, M_train, net, dim[0])
    
def main_generate_trainings_data():
    for rank in range(5,6):
        print(rank)
        dim = 25
        # rank = 2
        sparsity = 0.95
        n_samples = 5000000
        
        data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
        t = time.time()
        U, L, S, M, M_tri = data_generator.generate_set(n_matrices=n_samples, do_pickle=True)
        print(time.time()-t)


if __name__ == '__main__':
    # main()
    # main_SVD()
    main_generate_trainings_data()
    
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      