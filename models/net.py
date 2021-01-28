import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from tensorflow.keras import layers, Sequential, Input, Model, initializers
from tensorflow.keras.layers import Reshape, Dense, Flatten
import utilities.metrics
from models.utilities import custom_loss, custom_loss_UV
from utilities.metrics import MatrixSparsity
from utilities.plot import plot_matrices


# #%%
# import sys

# sys.path.append('/Users/ericbrunner/git/denise/')
# print(sys.path)

# #%%

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
        self._create_network()

        self.history = None

    def _create_network(self):
        self.model = Sequential()
        self.model.add(layers.Dense(self.dim // 2, activation="relu"))
        self.model.add(layers.Dense(self.dim // 2, activation="relu"))
        self.model.add(layers.Dense(self.dim // 2, activation="relu"))
        self.model.add(layers.Dense(self.output_dim[0] * self.output_dim[1]))
        self.model.add(layers.Reshape(self.output_dim))  # to matrix
        # model.add(Shrink(self.dim))

    def compile(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics
                           )

    def train(self, X, y):
        self.compile()

        self.history = self.model.fit(X, y,
                                      batch_size=self.batch_size,
                                      epochs=self.n_epochs,
                                      validation_split=self.valid_split,
                                      )
        self.model.summary()

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def load_weights(self, path):
        x = np.zeros((1, int(self.input_dim)))
        self.model(x)
        self.model.load_weights(path)

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


class Shrink(tf.keras.layers.Layer):
    def __init__(self, dim):
        super(Shrink, self).__init__()
        self.dim = float(dim)

    def call(self, inputs, **kwargs):
        return tf.multiply(tf.abs(inputs) - 1/tf.sqrt(self.dim), tf.sign(inputs))


class NeuralNet_SVD:
    def __init__(self, dim, rank, lr=0.001, batch_size=64, n_epochs=30, iterations=10, valid_split=0.2,
                 loss='mean_absolute_error', metrics=None):
        self.dim = dim # tuple (n,m)
        self.n = dim[0]
        self.m = dim[1]
        self.rank = rank
        self.output_dim_U = (self.n,self.rank)  # tuple (n, k)
        self.output_dim_V = (self.m,self.rank)  # tuple (m, k)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.iterations = iterations
        self.valid_split = valid_split

        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = custom_loss_UV(self.n)
        self.metrics = metrics
        self.model = self._create_network()

        self.history = None

    def _create_network(self):
        input_tensor = Input(shape=(self.n,self.m), name='Input')
        UV = Flatten()(input_tensor)
        U = Dense(2*self.n*self.m, activation='relu', name='U1')(UV)
        U = Dense(self.n*self.m//2, activation='relu',name='U2')(U)
        U = Dense(2*self.n*self.m, activation='relu',name='U3')(U)
        # print(U3.shape)
        U = Dense(self.output_dim_U[0] * self.output_dim_U[1], name='U4')(U)
        # print(U4.shape)
        U = Reshape(self.output_dim_U)(U)

        V = Dense(2*self.n*self.m, bias_initializer="zeros", activation='relu', name='V1')(UV)
        V = Dense(self.n*self.m//2, activation='relu',name='V2')(V)
        V = Dense(2*self.n*self.m, activation='relu',name='V3')(V)
        V = Dense(self.output_dim_V[0] * self.output_dim_V[1], name='V4')(V)
        V = Reshape(self.output_dim_V)(V)
        UV = tf.concat([U,V], axis=-2)
        model = Model(inputs=input_tensor, outputs=UV)
        return model

    def train(self, X, y):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics
                           )
        
        history = []
        for k in range(self.iterations):
            print('-----')
            print(f'Iteration {k}')
            print('-----')
            for x in ['U1','U2','U3','U4']:
                self.model.get_layer(x).trainable = True
            for x in ['V1','V2','V3','V4']:
                self.model.get_layer(x).trainable = False    
            history.append(
                self.model.fit(X, y,
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    validation_split=self.valid_split,
                    verbose=1
                    )
                )
            for x in ['U1','U2','U3','U4']:
                self.model.get_layer(x).trainable = False
            for x in ['V1','V2','V3','V4']:
                self.model.get_layer(x).trainable = True                
            history.append(
                self.model.fit(X, y,
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    validation_split=self.valid_split,
                    verbose=1
                    )
                )
        self.history = history
        self.model.summary()

    def plot_metrics(self, metrics):
        for metric in metrics:
            self._plot_one_metric(metric=metric)

    def _plot_one_metric(self, metric):
        train_metrics = [x for hist in self.history for x in hist.history[metric]]
        val_metrics = [x for hist in self.history for x in hist.history['val_' + metric]]
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
    
