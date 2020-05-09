import numpy as np
import matplotlib.pyplot as plt


class FullyConnected():
    def __init__(self, inputUnits, hiddenUnits, outputUnits, learningRate):
        # self.inputs = inputs.flatten()
        # self.targets = self.oneHotEncoding(targets)
        self.inputUnits = inputUnits
        self.hiddenUnits = hiddenUnits
        self.outputUnits = outputUnits
        self.learningRate = learningRate
        self.hiddenWeights = np.random.randn(
            self.inputUnits, self.hiddenUnits) / self.inputUnits
        self.outputWeights = np.random.randn(
            self.hiddenUnits, self.outputUnits)

        self.hiddenBias = np.zeros(shape=(1, self.hiddenUnits))
        self.outputBias = np.zeros(shape=(1, self.outputUnits))

    def softmax(self, input):
        input /= np.max(input)
        return np.exp(input) / np.sum(np.exp(input))

    def softmaxBackProp(self, dOut):
        grad, i = dOut

        totalExp = (np.exp(self.lastTotal)).flatten()
        S = np.sum(totalExp)

        deltaOutputTotal = -totalExp * totalExp / (S**2)
        # import pdb
        # pdb.set_trace()
        deltaOutputTotal[i] = totalExp[i] * (S - totalExp[i]) / (S**2)
        # Gradients of totals against weights/biases/input
        deltaOutputWeight = self.lastHiddenInputs
        deltaOutputBias = 1
        deltaOutputInput = self.outputWeights

        # Gradients of loss against totals
        deltaLossOutput = grad * deltaOutputTotal
        deltaLossOutput = deltaLossOutput.reshape(1, self.numClass)

        # Gradients of loss against weights/biases/input
        deltaLossWeights = deltaOutputWeight.T @ deltaLossOutput
        deltaLossBias = deltaLossOutput * deltaOutputBias
        deltaLossInputs = self.outputWeights @ deltaLossOutput.T

        # Update weights / biases
        # self.weights -= self.learningRate * deltaLossWeights
        self.outputBias += self.learningRate * deltaLossBias
        self.outputWeights += deltaLossWeights * self.learningRate

        return deltaLossInputs
    # .reshape(self.last_input_shape)

    def costFunction(self, targets, prediction):
        '''Cross Entropy Log-Loss'''
        return -np.log(prediction.flatten()[targets])

    def activation(self, X):
        '''RELU Activation'''
        return np.maximum(0, X)

    def activationDerivative(self, X):
        '''RELU Activation'''
        X[X > 0] = 1
        X[X <= 0] = 0
        return X

    def oneHotEncoding(self, labels):
        # numClass = len(np.unique(labels))
        self.numClass = 10
        y_enc = []
        # for num in labels:
        row = np.zeros(self.numClass)
        row[labels] = 1
        # y_enc.append(row)
        # y_enc = np.array(y_enc)
        return row

    def forwardPass(self, inputs):

        a1 = np.dot(inputs, self.hiddenWeights) + self.hiddenBias
        h1 = self.activation(a1)
        self.lastHiddenInputs = h1
        a2 = np.dot(h1, self.outputWeights) + self.outputBias
        self.lastTotal = a2
        h2 = self.softmax(a2)
        return h1, h2

    def backProp(self, prediction, h1, gradient, inputs):
        # outputError = (self.targets - prediction)

        # outputDelta = outputError * self.activationDerivative(prediction)
        # h1Error = np.dot(outputDelta, self.outputWeights.T)
        # h1Delta = h1Error * self.activationDerivative(h1)
        # h1Delta
        h1Error = self.softmaxBackProp(dOut=gradient)
        h1Delta = h1Error * (self.activationDerivative(h1)
                             ).reshape(self.hiddenUnits, 1)
        inputs = inputs.reshape(self.inputUnits, 1)
        self.hiddenWeights += np.dot(inputs,
                                     h1Delta.T) * self.learningRate
        # self.outputWeights += np.dot(h1.T, outputDelta) * self.learningRate

        self.hiddenBias += np.sum(h1Delta, axis=0) * self.learningRate
        # self.outputBias += np.sum(outputDelta, axis=0) * self.learningRate

    def run(self, epochs, inputs, targets):
        # errorVector = []
        # for _ in range(epochs):
        self.lastInputsShape = inputs.shape
        inputs = inputs.flatten()
        self.lastInputs = inputs
        h1, pred = self.forwardPass(inputs)
        cost = self.costFunction(targets, pred)

        prediction = pred.flatten()
        gradient = -1 / prediction[targets], targets

        # self.backProp(prediction, h1, gradient, inputs)
        # error = prediction - targets
        # error = np.sum(error**2)
        # # errorVector.append(error)

        # print(output)
        return cost, pred

    def predict(self, X):
        return self.forwardPass(X)


if __name__ == "__main__":

    inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    targets = np.array([[0], [0], [1], [1]])
    layer = FullyConnected(inputs=inputs, targets=targets, inputUnits=2,
                           hiddenUnits=2, outputUnits=1, learningRate=0.1)
    layer.run(6000)
    # plt.show()
