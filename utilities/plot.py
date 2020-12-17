import matplotlib.pyplot as plt
import tensorflow as tf


def plot_matrices(M_test, U_pred):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3, 8))
    L_pred = tf.matmul(U_pred, tf.transpose(U_pred))
    S_pred = M_test - L_pred
    axes[0].imshow(M_test)
    axes[0].set_ylabel('M    ', fontsize='x-large', rotation=0)
    axes[1].imshow(L_pred)
    axes[1].set_ylabel('L    ', fontsize='x-large', rotation=0)
    axes[2].imshow(S_pred)
    axes[2].set_ylabel('S    ', fontsize='x-large', rotation=0)

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
