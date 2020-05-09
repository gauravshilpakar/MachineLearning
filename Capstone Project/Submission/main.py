from dense_layer import FullyConnected
from conv_layer import Conv2D
from maxpool import Maxpool2D
from model import Model
from utils import create_dataset, onehot_encoder, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time
import matplotlib.pyplot as plt

'''
Input Dimensions 
N = Batch, C = channel, H = Height , W = Width
training data X = (N,C,H,W)
K = No. of Classes, N = Batch
training data Y = (K,N) (one hot encoded)

Output Dimensions
K = No. of Classes, N = Batch
Y_pred = (K,N)
'''

# Hyperparameters
learning_rate = 1e-3
epochs = 20
batch_size = 32

X_train, y_train, X_test, y_test = create_dataset()


X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]

y_train = onehot_encoder(y_train)

X_train = X_train/255
X_test = X_test/255

# Model
conv = Conv2D(8, (3, 3), lr=learning_rate)
pool = Maxpool2D(pool=(2, 2), stride=(2, 2))
dense = FullyConnected(layers=[32, 2], lr=learning_rate)

layers = [conv, pool, dense]

model = Model(layers)
print(f'Started Training for {epochs} epochs with batch size {batch_size}')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

t0 = time.time()
y_pred = model.predict(X_test)
t1 = time.time()

y_pred = np.argmax(y_pred, axis=0)
acc_score = accuracy_score(y_test, y_pred)
print(f'test_acc: {acc_score}, testing time: {round(t1-t0,4)} sec')


plt.show()
