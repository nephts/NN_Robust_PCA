import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np

def custom_loss(M_true, U_pred):
    """
    L1 function of U * U.T - M
    """
    U_pred_t = K.permute_dimensions(U_pred, pattern=(0, 2, 1))  # U (batch) transposed
    return K.mean(K.abs(K.batch_dot(U_pred, U_pred_t) - M_true))

    # L_pred = K.batch_dot(U_pred, U_pred_t)
    # n = tf.norm(L_pred - M_true, ord=1, axis=[-2, -1])
    # return n

def custom_loss_UV(n):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(M_true, UV_pred):
        U = UV_pred[:,:n]
        V_t = K.permute_dimensions(UV_pred[:,n:], pattern=(0, 2, 1))  # V (batch) transposed
        return K.mean(K.abs(K.batch_dot(U, V_t) - M_true))
   
    # Return a function
    return loss

def get_eigenvalues_and_vectors(U,V=None):
    e_val_U = np.linalg.norm(U, axis=0)**2
    e_vec_U = U / (e_val_U)**0.5
    if V is not None:
        e_val_V = np.linalg.norm(V, axis=0)**2
        e_vec_V = U / (e_val_V)**0.5
        return e_val_U, e_val_V, e_vec_U, e_vec_V
    return e_val_U, e_vec_U






