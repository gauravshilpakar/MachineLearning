from PartA import FeedForwardNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetworkTangent(FeedForwardNeuralNetwork):
    def activation(self, Z):
        return np.tanh(Z)

    def activationDerivative(self, Z):
        return 1 - np.square(Z)

    def costFunction(self, targets, prediction):
        return (1 / len(targets)) * np.sum(np.square(targets - prediction))


if __name__ == "__main__":
    np.random.seed(103)
    X = 2 * np.random.uniform(0, 1, 50) - 1
    T = np.sin(2 * np.pi * X) + 0.3 * np.random.normal(size=(1, 50))
    X = X.reshape(50, 1)
    T = T.reshape(50, 1)

    modelThree = NeuralNetworkTangent(inputUnits=1, hiddenUnits=3,
                                      outputUnits=1, epochs=5000, lr=0.1)
    predThree, _, costVectorThree = modelThree.run(X / np.max(X), T)
    modelTwenty = NeuralNetworkTangent(inputUnits=1, hiddenUnits=20,
                                       outputUnits=1, epochs=5000, lr=0.1)

    predTwenty, _, costVectorTwenty = modelTwenty.run(X / np.max(X), T)

    X = X.reshape(1, 50)
    T = T.reshape(1, 50)
    predThree = predThree.reshape(1, 50)
    predTwenty = predTwenty.reshape(1, 50)

    plt.subplot(2, 2, 1)

    plt.semilogy(costVectorThree)
    plt.title('Three Hidden Units')
    plt.xlabel('Epochs')
    plt.ylabel('Semilog Loss')
    plt.subplot(2, 2, 2)
    plt.semilogy(costVectorTwenty)
    plt.title('Twenty Hidden Units')
    plt.xlabel('Epochs')
    plt.ylabel('Semilog Loss')
    plt.subplot(2, 2, 3)
    plt.scatter(X, T)
    plt.scatter(X, predThree)
    plt.legend(['Data', 'Predicted Model'])
    plt.subplot(2, 2, 4)
    plt.scatter(X, T)
    plt.scatter(X, predTwenty)
    plt.legend(['Data', 'Predicted Model'])

    plt.show()
