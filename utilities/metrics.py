import tensorflow as tf
import keras.backend as K


class MatrixSparsity(tf.keras.metrics.Metric):
    def __init__(self, dim, eps=0.01, name='sparsity', **kwargs):
        super(MatrixSparsity, self).__init__(name=name, **kwargs)
        self.sparsity = 0
        self.size = dim ** 2
        self.eps = eps

    def update_state(self, M_true, U_pred, sample_weight=None):
        # Compute L = UU^T
        U_pred_t = K.permute_dimensions(U_pred, pattern=(0, 2, 1))
        L_pred = K.batch_dot(U_pred, U_pred_t)

        # Compute S = M - L
        S = M_true - L_pred

        # Get number of values e_ij in S with |e_ij| < eps
        n = K.cast(K.greater(self.eps, tf.abs(S)), 'float32')
        # Get mean over batch
        self.sparsity = tf.divide(K.sum(tf.math.count_nonzero(n, axis=(1, 2))), self.size)

    def result(self):
        return self.sparsity

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.sparsity = 0


class MatrixRank(tf.keras.metrics.Metric):
    def __init__(self, dim, eps=0.01, name='rank', **kwargs):
        super(MatrixRank, self).__init__(name=name, **kwargs)
        self.rank = 0
        self.size = dim ** 2
        self.eps = eps

    def update_state(self, M_true, U_pred, sample_weight=None):
        # Compute L = UU^T
        U_pred_t = K.permute_dimensions(U_pred, pattern=(0, 2, 1))
        L_pred = K.batch_dot(U_pred, U_pred_t)

        # Compute eigenvalues of L
        eigenvalues = tf.linalg.eigvalsh(L_pred)
        # Get all eigenvalues, which are > self.eps
        n_valid_eigenvalues = tf.math.count_nonzero(K.cast(K.greater(eigenvalues, self.eps), 'float32'), axis=1)
        # Get mean over batch
        self.rank = tf.divide(K.sum(n_valid_eigenvalues), self.size)

    def result(self):
        return self.rank

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.rank = 0
