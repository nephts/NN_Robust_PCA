import matplotlib.pyplot as plt
from tensorflow.keras import layers, Sequential


class NeuralNet:
    def __init__(self, input_dim, output_dim, batch_size=64, n_epochs=30, valid_split=0.2,
                 optimizer='adam', loss='mean_absolute_error', metrics=None):
        self.input_dim = input_dim  # n * (n+1) / 2
        self.output_dim = output_dim  # tuple (n, k)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.valid_split = valid_split

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = self._create_network()

        self.history = None

    def _create_network(self):
        model = Sequential()
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
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
                                      validation_split=self.valid_split
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
        plt.show()

    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test, verbose=0)
        return score

