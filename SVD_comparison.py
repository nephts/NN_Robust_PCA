#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:44:37 2021

@author: ericbrunner
"""

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

dim = [15,25]
rank = 7
n_samples = 1000000
n_epochs = 5
iterations = 5

psychdata = psych.Psychdata()
# data = psychdata.get_corr()
# print(data)

data = psychdata.get_reduced_data()

data_np = data.to_numpy()
data_np_red = np.delete(data_np[:17,:], 11, 0)
data_np_red = np.delete(data_np_red, 8, 0)
# data.drop(['Unnamed: 0', 'education', 'gender', 'age'], axis=1, inplace=True)
                      
u, s, vhh = np.linalg.svd(data_np_red, full_matrices=False)   

net = NeuralNet_SVD(rank=rank, n_epochs=n_epochs, iterations=iterations, dim=dim, batch_size=64,
                loss=custom_loss, metrics=[MatrixSparsity_SVD(dim=dim)])

weights_path = f'models/SVD_{dim}_{rank}_{n_samples}_{n_epochs}x{iterations}_weights.h5'
# x = np.zeros((1, int(self.n), int(self.m)))
# net(x)
net.model.load_weights(weights_path)

net.model.summary()




#%%
path = os.path.dirname(os.path.abspath(__file__))
n_epochs = 1
iterations = 1
n_samples = 1000000
dim = [15,25]
rank = 7
sparsity = 0.95
nk = int(n_samples/1000)
M_set = pickle.load( open( path + '/data/synthetic_matrices/SVD_M_dim'+str(dim)+'_rank'+str(rank)+'_n'+str(nk)+'k.p', 'rb' ) )
# index = np.random.randint(length)
index = 999998
UV_pred = net.predict(M_set[index:index+1])
U_pred = UV_pred[:,:dim[0]]
V_pred = UV_pred[:,dim[0]:]
L_pred = U_pred[0] @ V_pred[0].transpose()
plot_matrices(M_set[index], L_pred=L_pred)


#%%

%timeit net.predict(M_set[index:index+1])


#%%

n = 10
dim = 10
n_epochs = 100
rank = 3
input_dim = dim * (dim + 1) / 2

print(32*n*(n+1)/2)

net2 = NeuralNet(n_epochs=n_epochs, dim=dim, output_dim=(dim, rank), batch_size=64,
                    loss=custom_loss, metrics=[MatrixSparsity(dim=dim)])

x = np.zeros((1, int(input_dim)))
net2.model(x)
net2.model.summary()

