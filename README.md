# Neural network robust principal component analysis
Denise: Deep Learning based Robust PCA for Positive Semidefinite Matrices
Python implementation of Denise from https://arxiv.org/pdf/2004.13612.pdf

We study a neural network ansatz for robust principal component analysis (RPCA). A neural network is trained to learn the mapping of positive semi-definite (PSD) matrices to their robust eigenvalue decomposition. This approach is further generalized to robust singular value decomposition (RSVD) of arbitrary matrices.

Model and training data generation can be found in `Code`. For comparison, the principal component pursuit algorithm (standard algorithm for RPCA) is implemented. A detailed description of the project and discussion of results can be found in `report`. The neural network approach is compared to the benchmark principal component pursuit algorithm.

### Dependencies:
- [NumPy](https://numpy.org/)  
- [Tensorflow](https://www.tensorflow.org/)  

Eric Brunner (eric.brunner@physik.uni-freiburg.de)
