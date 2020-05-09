import numpy as np


class Softmax():
    def forwardPass(self, input):
        sm = np.exp(input) / np.sum(np.exp(input), axis=0)

        return
