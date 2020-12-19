import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential
from utilities.metrics import MatrixSparsity
from utilities.plot import plot_matrices


class NeuralNet:
    def __init__(self, dim, output_dim, lr=0.001, batch_size=64, n_epochs=30, valid_split=0.2,
                 loss='mean_absolute_error', metrics=None):
        self.dim = dim
        self.input_dim = dim * (dim + 1) / 2  # n * (n+1) / 2
        self.output_dim = output_dim  # tuple (n, k)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.valid_split = valid_split

        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = loss
        self.metrics = metrics
        self.model = self._create_network()

        self.history = None

    def _create_network(self):
        model = Sequential()
        model.add(layers.Dense(self.dim // 2, activation="relu"))
        model.add(layers.Dense(self.dim // 2, activation="relu"))
        model.add(layers.Dense(self.dim // 2, activation="relu"))
        model.add(layers.Dense(self.output_dim[0] * self.output_dim[1]))
        model.add(layers.Reshape(self.output_dim))  # to matrix

        return model

    def train(self, X, y):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics
                           )

        self.history = self.model.fit(X, y,
                                      batch_size=self.batch_size,
                                      epochs=self.n_epochs,
                                      validation_split=self.valid_split,
                                      )
        self.model.summary()

    def plot_metrics(self, metrics):
        for metric in metrics:
            self._plot_one_metric(metric=metric)

    def _plot_one_metric(self, metric):
        train_metrics = self.history.history[metric]
        val_metrics = self.history.history['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.savefig(f'plots/{metric}.pdf')
        plt.show()

    def evaluate(self, M_test, dim):
        if len(M_test.shape) == 2:
            M_test = M_test[None, :, :]

        # Convert
        M_tris = list()
        for m in M_test:
            tri_M = m[np.triu_indices(m.shape[0])]
            M_tris.append(tri_M)
        M_tris = np.array(M_tris)
        U_pred = self.predict(X=M_tris)

        matrix_sparsity = MatrixSparsity(dim=dim)
        matrix_sparsity.update_state(M_true=M_test, U_pred=U_pred)
        sparsity_test = matrix_sparsity.result()

        # Plot one example
        fig = plot_matrices(M_test[0], U_pred=U_pred[0])
        plt.savefig('plots/denise_output.pdf')
        fig.show()
        return sparsity_test, U_pred

    def predict(self, X):
        return self.model.predict(X)

