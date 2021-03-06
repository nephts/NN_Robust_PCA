#!/usr/bin/env python
import numpy as np
import pickle
import os


class SyntheticMatrixSet:
    """
    Class generates a nxn positive semidefinite matrices  M = L + S with for a known matrix L
    of rank k≤n and a known matrix S of given sparsity s.

        L:
            L.1. Create nxk matrix U with elements ~ N(0,1)
            L.2. L = U.Transpose(U)

        Sprop:
            Sprop.1. select uniformly randomly pair (i,j), with 1≤i<j≤n
            Sprop.2. select uniformly randomly number b in [-1,1] S(i,j) = S(j,i) = b.
            Sprop.3. S(i,i) = S(j,j) = uniformly number of [|b|,1].

        S: 	um up Sprop1 + Sprop2 until the sparsity s is reached


        Description of the methods

        generateL(dim, rank, sparsity) returns L of given rank

        generateS(dim, rank, sparsity) returns S of given sparsity

        generate(dim, rank, sparsity)  returns M, L(rank k), S
    """

    def __init__(self, dim, rank, sparsity):
        self.dim = dim
        self.rank = rank
        self.sparsity = sparsity

        self.U_set = list()
        self.L_set = list()
        self.S_set = list()
        self.M_set = list()
        self.M_tri_set = list()

    def _add(self, U, L, S, M, M_tri):
        self.U_set.append(U)
        self.L_set.append(L)
        self.S_set.append(S)
        self.M_set.append(M)
        self.M_tri_set.append(M_tri)

    def generateU(self):
        # L.1.
        U = np.random.randn(self.dim, self.rank)
        # L.2.
        return U

    def generateSprop(self):
        # Sprop.1.
        indexpair = np.random.randint(self.dim, size=2)
        while indexpair[0] == indexpair[1]:
            indexpair = np.random.randint(self.dim, size=2)

        # create empty nxn matrix
        S = np.zeros((self.dim, self.dim))

        # Sprop.2.
        b = 2 * np.random.rand() - 1
        S[indexpair[0], indexpair[1]] = b
        S[indexpair[1], indexpair[0]] = b

        # Sprop.3.
        c = (1 - abs(b)) * np.random.rand() + abs(b)
        S[indexpair[0], indexpair[0]] = c
        S[indexpair[1], indexpair[1]] = c

        return S

    def generateS(self):
        S = self.generateSprop()
        s = (S.size - np.count_nonzero(S)) / (S.size)

        while s > self.sparsity:
            S += self.generateSprop()
            s = (S.size - np.count_nonzero(S)) / (S.size)

        return S

    def generate(self):
        U = self.generateU()
        L = np.matmul(U, U.transpose())
        S = self.generateS()
        M = L + S
        M_tri = self._get_upper_triangle(M)
        return U, L, S, M, M_tri

    def generate_set(self, n_matrices, do_pickle=False):
        for _ in range(n_matrices):
            U, L, S, M, M_tri = self.generate()
            self._add(U=U, L=L, S=S, M=M, M_tri=M_tri)
        U_set = np.array(self.U_set)
        L_set = np.array(self.L_set)
        S_set = np.array(self.S_set)
        M_set = np.array(self.M_set)
        M_tri_set = np.array(self.M_tri_set)
        if do_pickle==True:
            N = int(n_matrices/1000)
            path = os.path.dirname(os.path.abspath(__file__))
            pickle.dump(U_set, open( path + '/synthetic_matrices/U_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(L_set, open( path + '/synthetic_matrices/L_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(S_set, open( path + '/synthetic_matrices/S_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(M_set, open( path + '/synthetic_matrices/M_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(M_tri_set, open( path + '/synthetic_matrices/M_tri_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
        return U_set, L_set, S_set, M_set, M_tri_set

    @staticmethod
    def _get_upper_triangle(arr):
        """
        Returns upper triangle part of the given matrix as vector.
        """
        return arr[np.triu_indices(arr.shape[0])]
    

class SyntheticMatrixSet_SVD:
    """
    Class generates a nxm matrix  M = L + S for a known matrix L
    of rank k≤n and a known matrix S of given sparsity s.

        L:
            L.1. Create nxk matrix U with elements ~ N(0,1)
            L.2. Create mxk matrix V with elements ~ N(0,1)

        Sprop:
            Sprop.1. select uniformly randomly pair (i,j), with 1≤i<j≤n
            Sprop.2. select uniformly randomly number b in [-1,1] S(i,j) = S(j,i) = b.
            Sprop.3. S(i,i) = S(j,j) = uniformly number of [|b|,1].

        S: 	um up Sprop1 + Sprop2 until the sparsity s is reached


        Description of the methods

        generateL(dim, rank, sparsity) returns L of given rank

        generateS(dim, rank, sparsity) returns S of given sparsity

        generate(dim, rank, sparsity)  returns M, L(rank k), S
    """

    def __init__(self, dim, rank, sparsity):
        self.dim = dim # tuple (n,m)
        self.n = dim[0]
        self.m = dim[1]
        self.rank = rank
        self.sparsity = sparsity

        self.U_set = list()
        self.V_set = list()
        self.L_set = list()
        self.S_set = list()
        self.M_set = list()

    def _add(self, U, V, L, S, M):
        self.U_set.append(U)
        self.V_set.append(V)
        self.L_set.append(L)
        self.S_set.append(S)
        self.M_set.append(M)

    def generate_UV(self):
        # L.1.
        U = np.random.randn(self.n, self.rank)
        # L.2.
        V = np.random.randn(self.m, self.rank)
        return U, V

    def generateS(self):
        S = np.zeros((self.n, self.m))
        s = (S.size - np.count_nonzero(S)) / (S.size)
        while s > self.sparsity:
            index1 = np.random.randint(self.n)
            index2 = np.random.randint(self.m)
            S[index1,index2] = 2 * np.random.rand() - 1
            s = (S.size - np.count_nonzero(S)) / (S.size)
        return S

    def generate(self):
        U,V = self.generate_UV()
        L = np.matmul(U, V.transpose())
        S = self.generateS()
        M = L + S
        return U, V, L, S, M

    def generate_set(self, n_matrices, do_pickle=False):
        for _ in range(n_matrices):
            U, V, L, S, M = self.generate()
            self._add(U=U, V=V, L=L, S=S, M=M)
        U_set = np.array(self.U_set)
        V_set = np.array(self.V_set)
        L_set = np.array(self.L_set)
        S_set = np.array(self.S_set)
        M_set = np.array(self.M_set)
        if do_pickle==True:
            N = int(n_matrices/1000)
            path = os.path.dirname(os.path.abspath(__file__))
            pickle.dump(U_set, open( path + '/synthetic_matrices/SVD_U_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(V_set, open( path + '/synthetic_matrices/SVD_V_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(L_set, open( path + '/synthetic_matrices/SVD_L_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(S_set, open( path + '/synthetic_matrices/SVD_S_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
            pickle.dump(M_set, open( path + '/synthetic_matrices/SVD_M_dim'+str(self.dim)+'_rank'+str(self.rank)+'_n'+str(N)+'k.p', 'wb' ) )
        return U_set, V_set, L_set, S_set, M_set

    
