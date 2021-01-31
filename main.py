import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import time
import os
import pickle

from models.net import NeuralNet, NeuralNet_SVD
from models.utilities import custom_loss
from utilities.metrics import MatrixSparsity, MatrixSparsity_SVD
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


def comparison(denise, method, data, dim, M_test, n_epochs, n_samples, test_set_size, rank, path, i=0):
    # DENISE
    score, U_denise = denise.evaluate(M_test=data, dim=dim, path=path, i=i)
    L_denise = np.matmul(U_denise[0], U_denise[0].T)
    S_denise = data - L_denise
    tf.print(f'Sparsity (DENISE): {score}')

    if method == 'pcp':
        L_method, S_method, (u_method, s_method, v_method) = pcp(M=data, mu=0.055)
    matrix_sparsity = MatrixSparsity(dim=dim)
    matrix_sparsity.update_state(M_true=M_test, L_pred=L_method)
    score_pcp = matrix_sparsity.result()
    tf.print(f'Sparsity (PCP): {score_pcp}')

    with open(f"{path}/res_{i}.txt", "w") as output:
        output.write(f"DENISE trained {n_epochs} epochs with {n_samples - test_set_size} {dim}x{dim} rank {rank} "
                     f"matrices.\nDENISE: \n\nL: \n {str(L_denise)}\n\nS: \n{str(S_denise)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\n{method}: \n\nL: \n{str(L_method)}\n\nS: \n{str(S_method)}\n\n")
        output.write("-" * 12)
        output.write(f"\n\nM: \n{str(data)}\n\n")
        output.write(f"Relative Error of L : {tf.divide(tf.norm(L_denise - L_method, ord='fro', axis=(0, 1)), tf.norm(L_method, ord='fro', axis=(0, 1)))}\n")
        output.write(f"Relative Error of S: {tf.divide(tf.norm(S_denise - S_method, ord='fro', axis=(0, 1)), tf.norm(S_method, ord='fro', axis=(0, 1)))}\n")

    fig = plot_matrices(data, L_pred=L_method) #, vmin=-1, vmax=1.1)
    plt.savefig(f'{path}{method}_{i}.pdf')
    fig.show()





def main():   
    
    # Load data --------------------------------------------------------------
    path = os.path.dirname(os.path.abspath(__file__))
    n_samples = 1000000
    n_epochs = 300
    dim = 10
    rank = 3
    load_weights = True
    save_weights = False
    weights_path = f'models/{dim}_{rank}_{n_samples}_{n_epochs}_weights.h5'
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
    # dim = 10
    # rank = 3
    # sparsity = 0.95
    # U, L, S, M, M_tri = get_data(data=data, dim=dim, rank=rank, sparsity=sparsity, n_samples=n_samples)
    
    # # Split data set ---------------------------------------------------------
    # test_set_size = int(0.2 * n_samples)
    # U_test, L_test, S_test, M_test, M_tri_test = U[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size], M_tri[:test_set_size]
    # U_train, L_train, S_train, M_train, M_tri_train = U[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:], M_tri[test_set_size:]

    # Network
    net = NeuralNet(n_epochs=n_epochs, dim=dim, output_dim=(dim, rank), batch_size=64,
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim)])

    # Train or load weights
    if not load_weights:
        print(f'starting training for {n_epochs} epochs on {n_samples} matrices...')
        net.train(X=M_tri_train, y=M_train)
        net.plot_metrics(metrics=['loss', 'sparsity'])
    else:
        print(f'load stored network trained for {n_epochs} epochs on {n_samples} matrices...')
        net.load_weights(weights_path)
    if save_weights:
        net.save_weights(weights_path)


    # Compare
    # ''' !!! UNCOMMENT the following code for compare on psych data !!! '''
    # # Compare on psychdata
    # psychdata = psych.Psychdata()
    # data = psychdata.get_corr()
    # comparison(denise=net, method='pcp', data=data, dim=dim, M_test=M_test, n_epochs=n_epochs,
    #          n_samples=n_samples, test_set_size=test_set_size, rank=rank)

    # ''' !!! UNCOMMENT the following code for compare on finance data !!! '''
    # Compare on Finance data
    current_path = os.getcwd()
    with open(current_path + '/Stock prices dax 30_new.csv') as stockprices:
        data = list(csv.reader(stockprices, delimiter=","))
    data_array = np.array(data)
    data_only = data_array[1:data_array.shape[0], 1:11].T
    data_only = np.array([[float(y) for y in x] for x in data_only])

    t = 35  # length of retrospective observation period

    Sigma = np.zeros((10, 10, data_only.shape[1] - t + 1))
    Corr = np.zeros((10, 10, data_only.shape[1] - t + 1))

    for l in range(0, data_only.shape[1] - t):

        Var = np.zeros(10)

        for k in range(l, l + t - 1):
            Sigma[:, :, l] = Sigma[:, :, l] + np.dot(
                np.subtract(data_only[:, k], np.mean(data_only[:, l:l + t - 1], axis=1)).reshape((10, 1)),
                np.subtract(data_only[:, k], np.mean(data_only[:, l:l + t - 1], axis=1)).reshape((1, 10)))
            Var = Var + np.subtract(data_only[:, k], np.mean(data_only[:, l:l + t - 1], axis=1)) ** 2

            # Sigma[:,:,l] = (1/t-1) * Sigma[:,:,l] aktivate for covariance matrix

        Corr[:, :, l] = Sigma[:, :, l] / np.sqrt(np.dot(Var.reshape((10, 1)), Var.reshape((1, 10))))

    for i in range(10):
        comparison(denise=net, method='pcp', data=Corr[:, :, i], dim=dim, M_test=M_test, n_epochs=n_epochs,
                    n_samples=n_samples, test_set_size=test_set_size, rank=rank, path='plots/finance_fixed_colorcode/', i=i)


def main_SVD():
    # Load data --------------------------------------------------------------
    # path = os.path.dirname(os.path.abspath(__file__))
    # n_epochs = 1
    # iterations = 1
    # n_samples = 1000000
    # dim = [15,25]
    # rank = 7
    # sparsity = 0.95
    # nk = int(n_samples/1000)
    # M = pickle.load( open( path + '/data/synthetic_matrices/SVD_M_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )

    # U = pickle.load( open( path + '/data/synthetic_matrices/SVD_U_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )
    # V = pickle.load( open( path + '/data/synthetic_matrices/SVD_V_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )
    # S = pickle.load( open( path + '/data/synthetic_matrices/SVD_S_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )

    # Split data set ---------------------------------------------------------
    # test_set_size = int(0.2 * n_samples)
    # M_test = M[:test_set_size]
    # M_train = M[test_set_size:]
    # S_test, M_test, U_test, V_test = S[:test_set_size], M[:test_set_size], U[:test_set_size], V[:test_set_size]
    # S_train, M_train, U_train, V_train = S[test_set_size:], M[test_set_size:], U[test_set_size:], V[test_set_size:]
    
    # Generate data ----------------------------------------------------------
    data = 'synthetic_SVD'
    n_epochs = 5
    iterations = 20
    n_samples = 100000
    dim = (5,4)
    rank = 2
    sparsity = 0.95
    
    test_set_size = int(0.2 * n_samples)
    
    U, V, L, S, M = get_data(data=data, dim=dim, rank=rank, sparsity=sparsity, n_samples=n_samples)
    
    # Split data set ---------------------------------------------------------
    U_test, V_test, L_test, S_test, M_test = U[:test_set_size], V[:test_set_size], L[:test_set_size], S[:test_set_size], M[:test_set_size]
    U_train, V_train, L_train, S_train, M_train = U[test_set_size:], V[test_set_size:], L[test_set_size:], S[test_set_size:], M[test_set_size:]
    

    net = NeuralNet_SVD(rank=rank, n_epochs=n_epochs, iterations=iterations, dim=dim, batch_size=64,
                    loss=custom_loss, metrics=[MatrixSparsity_SVD(dim=dim)])
    

    # Train or load weights
    load_weights = True
    save_weights = False
    weights_path = f'models/SVD_{dim}_{rank}_{n_samples}_{n_epochs}x{iterations}_weights.h5'
    if not load_weights:
        print(f'starting training for {n_epochs} epochs on {n_samples} matrices...')
        net.train(X=M_train, y=M_train)
        net.plot_metrics(metrics=['loss', 'sparsity'])
    else:
        print(f'load stored network trained for {n_epochs} epochs on {n_samples} matrices...')
        net.load_weights(weights_path)
    if save_weights:
        net.save_weights(weights_path)
    
    test_UV(U_test, V_test, M_test, net, dim[0])
    test_UV(U_test, V_test, M_test, net, dim[0])
    test_UV(U_test, V_test, M_test, net, dim[0])
    
def main_generate_trainings_data():
    
    rank = 7
    # rank = 2
    sparsity = 0.95
    n_samples = 1000000

    # dim = 25
    # data_generator = SyntheticMatrixSet(dim=dim, rank=rank, sparsity=sparsity)
    # t = time.time()
    # U, L, S, M, M_tri = data_generator.generate_set(n_matrices=n_samples, do_pickle=True)
    # print(time.time()-t)
    
    dim = [15,25]
    data_generator = SyntheticMatrixSet_SVD(dim=dim, rank=rank, sparsity=sparsity)
    t = time.time()
    U, V, L, S, M = data_generator.generate_set(n_matrices=n_samples, do_pickle=True)
    print(time.time()-t)


if __name__ == '__main__':
    main()
    # main_SVD()
    # main_generate_trainings_data()
    
                      
                      
               


       