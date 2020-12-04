from models.net import NeuralNet
from data.generation import DataGenerator

if __name__ == '__main__':
    n_samples = 20
    dim = 10

    data_generator = DataGenerator(size=n_samples, dim=dim)
    X, y = data_generator.generate()

    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

    net = NeuralNet(input_dim=dim, output_dim=5)
    net.train(X=X, y=y)
    net.plot_metric(metric='loss')
