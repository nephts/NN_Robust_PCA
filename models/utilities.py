import keras.backend as K
import numpy as np


def custom_loss(M_true, U_pred):
    """
    L1 function of U * U.T - M
    """
    # reshape vector to matrix
    # U (batch) transposed
    U_pred_t = K.permute_dimensions(U_pred, pattern=(0, 2, 1))
    return K.sum(K.abs(K.batch_dot(U_pred, U_pred_t) - M_true))
