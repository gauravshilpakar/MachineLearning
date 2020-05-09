import numpy as np


class Conv2D():
    def __init__(self, numFilters):
        self.numFilters = numFilters
        self.filters = np.random.randn(self.numFilters, 3, 3) / 9
        # Divided by 9 as per XAVIER INITIALIZATION

    def regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                imageRegion = image[i:(i + 3), j:(j + 3)]
                yield imageRegion, i, j

    def forwardPass(self, input):
        h, w = input.shape
        a = np.zeros((h - 2, w - 2, self.numFilters))

        for imageRegion, i, j in self.regions(input):
            a[i, j] = np.sum(imageRegion * self.filters, axis=(1, 2))

        return a


if __name__ == "__main__":
    layer = Conv2D(numFilters=3)
    a = layer.forwardPass(np.ones(shape=(3, 3)))

    print(layer.filters)
    print()
    print(a)
