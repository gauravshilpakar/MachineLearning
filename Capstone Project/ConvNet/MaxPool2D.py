import numpy as np


class MaxPool2D():
    def regions(self, image):
        h, w, numFilters = image.shape
        h = h // 2
        w = w // 2

        for i in range(h - 2):
            for j in range(w - 2):
                imageRegion = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield imageRegion, i, j

    def forwardPass(self, input):
        h, w, numFilters = input.shape
        a = np.zeros((h // 2, w // 2, numFilters))
        for imageRegion, i, j in self.regions(input):
            a[i, j] = np.amax(imageRegion, axis=(0, 1))

        return a
