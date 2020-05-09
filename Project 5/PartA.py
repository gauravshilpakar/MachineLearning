import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm


class FeedForwardNeuralNetwork():
    def __init__(self, inputUnits, hiddenUnits, outputUnits, lr=1e-4, epochs=300, hiddenWeights=None, hiddenBias=None):
        self.inputUnits = inputUnits
        self.hiddenUnits = hiddenUnits
        self.outputUnits = outputUnits
        self.lr = lr
        self.epochs = epochs
        self.hiddenWeights = np.random.uniform(
            size=(self.inputUnits, self.hiddenUnits))
        self.hiddenBias = np.zeros(shape=(1, self.hiddenUnits))

        self.outputWeights = np.random.uniform(
            size=(self.hiddenUnits, self.outputUnits))
        self.outputBias = np.zeros(shape=(1, self.outputUnits))

    def activation(self, Z):
        return 1 / (1 + np.exp(-Z))

    def activationDerivative(self, Z):
        return Z * (1 - Z)

    def costFunction(self, targets, prediction):
        return (-1 / len(targets)) * (np.sum(np.multiply(targets, np.log(prediction))))

    def predict(self, inputs):
        return self.forwardPass(inputs)

    def forwardPass(self, inputs):
        # HIDDEN LAYER
        hiddenOutput = np.dot(inputs, self.hiddenWeights) + self.hiddenBias
        hiddenOutput = self.activation(hiddenOutput)
        # OUTPUT LAYER
        outputZ = np.dot(
            hiddenOutput, self.outputWeights) + self.outputBias
        prediction = self.activation(outputZ)
        return hiddenOutput, prediction

    def run(self, inputs, targets):

        costVector = []
        cost = 1000
        ite = 0

        for _ in range(self.epochs):
            hiddenOutput, prediction = self.forwardPass(inputs)
            # BACKPROP
            error = (targets - prediction)

            deltaOutput = error * self.activationDerivative(prediction)
            errorHidden = np.dot(deltaOutput, self.outputWeights.T)
            deltaHidden = errorHidden * self.activationDerivative(hiddenOutput)

            self.outputWeights += np.dot(hiddenOutput.T, deltaOutput) * self.lr
            self.hiddenWeights += np.dot(inputs.T, deltaHidden) * self.lr
            self.hiddenBias += np.sum(deltaHidden, axis=0) * self.lr
            self.outputBias += np.sum(deltaOutput, axis=0) * self.lr

            error = np.sum(error**2)
            cost = self.costFunction(targets, prediction)
            costVector.append(cost)
            # print(f"{_}\t Cost: {round(cost,4):.6f}")
        return prediction, hiddenOutput, costVector


if __name__ == "__main__":
    inputs = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
    targets = np.array([[0], [0], [1], [1]])

    inputUnits = 2
    hiddenUnits = 2
    outputUnits = 1

    model = FeedForwardNeuralNetwork(
        inputUnits, hiddenUnits, outputUnits, lr=0.1, epochs=6000)

    pred, hidden, costVector = model.run(inputs, targets)
    outputPred = pred
    print(
        f"\nAfter {model.epochs} epochs, loss = {round(costVector[-1],4):.6f}")
    print(f"\nTargets\n{targets}")
    print(f"\nPredictions\n{pred}")

    _, pred = model.predict([[2, 2]])
    print(f"\nXOR of [[2, 2]]:{pred}\n")
    _, pred = model.predict([[-2, -2]])
    print(f"\nXOR of [[-2, -2]]:{pred}\n")
    _, pred = model.predict([[2, -2]])
    print(f"\nXOR of [[2, -2]]:{pred}\n")
    plt.figure()
    plt.semilogy(costVector)

    plt.title('XOR Classification')
    plt.xlabel('Epochs')
    plt.ylabel('Semilog Cross Entropy Loss')
    plt.tight_layout()

    X, Y = np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1))
    _, Z = model.predict(np.array([X.ravel(), Y.ravel()]).T)
    Z = Z.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    xs = []
    ys = []
    for _ in range(len(inputs)):
        x, y = inputs[_]
        xs.append(x)
        ys.append(y)
    ax.scatter(xs, ys, outputPred)
    plt.title('Decision Surface')
    plt.show()
