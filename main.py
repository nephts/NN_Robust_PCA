from models.net import NeuralNet
from data.generation import DataGenerator

if __name__ == '__main__':
    n_samples = 100
    dim = 10

    data_generator = DataGenerator(size=n_samples, dim=dim)
    X, y = data_generator.generate()

    net = NeuralNet(input_dim=dim, output_dim=5)
    net.train(X=X, y=y)
    net.plot_metric(metric='loss')
