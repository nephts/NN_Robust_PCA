import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_matrices(M_test, U_pred=None, L_pred=None):
    # if U_pred is None and L_pred is None:
    #     raise ValueError('U_pred and L_pred can not be None both')
    # if U_pred is not None:
    #     L_pred = tf.matmul(U_pred, tf.transpose(U_pred))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 8))
    S_pred = M_test - L_pred
    im = axes[0].imshow(M_test)
    axes[0].set_ylabel('M    ', fontsize='x-large', rotation=0)
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(L_pred)
    axes[1].set_ylabel('L    ', fontsize='x-large', rotation=0)
    fig.colorbar(im, ax=axes[1])
    
    # print(M_test)
    # print(L_pred)
    # print(S_pred)
    im = axes[2].imshow(S_pred)
    axes[2].set_ylabel('S    ', fontsize='x-large', rotation=0)
    fig.colorbar(im, ax=axes[2])

    for i in range(3):
        axes[i].tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
            right=False,
            left=False)
    return fig

def test_UV(U_set, V_set, M_set, net, n):
    length = len(U_set)
    index = np.random.randint(length)
    UV_pred = net.predict(M_set[index:index+1])
    U_pred = UV_pred[:,:n]
    V_pred = UV_pred[:,n:]
    L_pred = U_pred[0] @ V_pred[0].transpose()
    plot_matrices(M_set[index], L_pred=L_pred)
