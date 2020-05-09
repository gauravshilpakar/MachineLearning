import mnist
import numpy as np
from Conv2D import Conv2D
from FullyConnected import FullyConnected
from MaxPool2D import MaxPool2D

import matplotlib.pyplot as plt
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


def run():
    sh = train_images[0].shape
    ConvLayer = Conv2D(8)
    MaxPool = MaxPool2D()

    FC = FullyConnected(13 * 13 * 8, 128, 10, 0.001)
    loss = []
    acc = 0
    for i, (im, label) in enumerate(zip(test_images, test_labels)):
        # Do a forward pass.
        output = ConvLayer.forwardPass(im / 255)
        output = MaxPool.forwardPass(output)
        l, prediction = FC.run(1000, output, label)
        loss.append(l)
        a = 1 if np.argmax(prediction) == label else 0
        import pdb
        # pdb.set_trace()
        acc += a
        # Print stats every 100 steps.
        if i % 100 == 99:
            #     print(
            print('[Step %d] Average Loss %.3f | Accuracy %d%%' %
                  (i + 1, np.average(loss), np.average(acc)))
        # )
        loss = []
        acc = 0


if __name__ == "__main__":
    run()
