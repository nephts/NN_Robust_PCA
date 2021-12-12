import SyntheticMatrices as SM
import numpy as np

MatrixSet = SM.SyntheticMatrixSet(5,4,0.5)

L = MatrixSet.generateL()
S = MatrixSet.generateS()
M = L+S


print(L)
print('\n')
print(S)
print('\n')
print(M)

