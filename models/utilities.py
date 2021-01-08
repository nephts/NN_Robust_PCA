import tensorflow.keras.backend as K
import tensorflow as tf

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
    def loss(M_true,UV_pred):
        U = UV_pred[:,:n]
        V_t = K.permute_dimensions(UV_pred[:,n:], pattern=(0, 2, 1))  # V (batch) transposed
        return K.mean(K.abs(K.batch_dot(U, V_t) - M_true))
   
    # Return a function
    return loss

