import os

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# importing custom logistic regression model
from part1 import LogisticRegression_

# random shuffling of dataset


def shuffling_files(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# split dataset in train, validation and test set at 8:1:1


def data_split(data):
    n_samples = len(data)
    itrain, itest = int(0.8 * n_samples), int(0.2 * n_samples)
    train, test = data[:itrain], data[itrain:]
    return train, test


filepath = "./Dataset/"


print("Loading Worms...")
worm_images = np.array([io.imread(filepath + '/Worm/' + image, as_gray=True)
                        for image in os.listdir(filepath + '/Worm/')])
worm_label = np.array([1 for i in worm_images])

print("Loading No Worms...")
noworm_images = np.array([io.imread(filepath + '/NoWorm/' + image, as_gray=True)
                          for image in os.listdir(filepath + '/NoWorm/')])
noworm_label = np.array([0 for i in noworm_images])

temp_X_train = np.concatenate((worm_images, noworm_images))
y_train = np.concatenate((worm_label, noworm_label))

X_data, y_data = shuffling_files(temp_X_train, y_train)
X_train, X_test = data_split(X_data)
y_train, y_test = data_split(y_data)

# X_train = X_train/255
# X_test = X_test/255

x, y, z = X_train.shape
X_train = X_train.reshape(x, y * z)
a, b, c = X_test.shape
X_test = X_test.reshape(a, b * c)

print('Training...')
model = LogisticRegression_(X_train, y_train, X_test, y_test)
model.train()
plt.show()


LR = LogisticRegression(max_iter=500, solver='liblinear')
LR.fit(X_train, y_train)

print(LR.score(X_train, y_train))
# y_pred = model.predict(X_test)
# y_pred = np.argmax(y_pred, axis=1)

# print('acc_test: {}'.format(accuracy_score(y_test, y_pred)))
