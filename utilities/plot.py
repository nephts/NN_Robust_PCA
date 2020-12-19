import matplotlib.pyplot as plt
import tensorflow as tf


def plot_matrices(M_test, U_pred=None, L_pred=None):
    if U_pred is None and L_pred is None:
        raise ValueError('U_pred and L_pred can not be None both')
    if U_pred is not None:
        L_pred = tf.matmul(U_pred, tf.transpose(U_pred))
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 8))
    S_pred = M_test - L_pred
    im = axes[0].imshow(M_test)
    axes[0].set_ylabel('M    ', fontsize='x-large', rotation=0)
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(L_pred)
    axes[1].set_ylabel('L    ', fontsize='x-large', rotation=0)
    fig.colorbar(im, ax=axes[1])

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
