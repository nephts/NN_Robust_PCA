import keras.backend as K


def custom_loss(M_true, U_pred):
    """
    L1 function of U * U.T - M
    """
    U_pred_t = K.permute_dimensions(U_pred, pattern=(0, 2, 1))  # U (batch) transposed
    return K.mean(K.abs(K.batch_dot(U_pred, U_pred_t) - M_true))
