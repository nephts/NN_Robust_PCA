#!/usr/bin/env python

import numpy as np
'''
Class generates a nxn positive semidefinite matrices  M = L + S with for a known matrix L of rank k≤n and a known matrix S of given sparsity s.

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
'''

class SyntheticMatrixSet:
	
	def __init__(self, dim, rank, sparsity):
		self.dim = dim
		self.rank = rank
		self.sparsity = sparsity
		
		
	
	def generateL(self):
		# L.1.
		U = np.random.randn(self.dim, self.rank)
		# L.2.	
		return np.matmul(U,U.transpose())			
		
	def generateSprop(self):
		# Sprop.1.
		indexpair = np.random.randint(self.dim, size=2)
		while indexpair[0]==indexpair[1]:
			indexpair = np.random.randint(self.dim, size=2)
		
		
		# create empty nxn matrix
		S = np.zeros((self.dim,self.dim))
		
		# Sprop.2.			
		b = 2*np.random.rand()-1
		S[indexpair[0],indexpair[1]] = b
		S[indexpair[1],indexpair[0]] = b
		
		# Sprop.3.
		c = (1-abs(b))*np.random.rand() + abs(b)
		S[indexpair[0],indexpair[0]] = c
		S[indexpair[1],indexpair[1]] = c
		
		return S
		
	def generateS(self):
		S = self.generateSprop()
		s = (S.size - np.count_nonzero(S))/(S.size)
		
		while s > self.sparsity :
			S += self.generateSprop()
			s = (S.size - np.count_nonzero(S))/(S.size)
			
		return S
		

	def generate(self):
		L = self.generateL()
		S = self.generateS()
		M = L + S
		return [M, L ,S]

		
	
	
	
	
	
	
		
		
