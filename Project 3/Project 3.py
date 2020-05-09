import pdb
import matplotlib.pyplot as plt
import numpy as np
from math import asin, exp, pi, sin, sqrt

np.random.seed(100)
L = 100
N = 25
E = np.random.normal(scale=0.3)
s = 0.1


class Dataset():
    def train_data(self, N_TRAIN):
        X_TRAIN = np.sort(np.random.uniform(0, 1, N_TRAIN))
        t_TRAIN = np.sin(2*pi*X_TRAIN) + E
        g = np.linspace(0, 1, N)
        phi = []
        den = 2 * (s**2)

        for i in range(N_TRAIN):
            num = (X_TRAIN[i] - g)**2
            phi.append(np.exp(-1*num/den))

        phi = np.array(phi)
        phi = np.concatenate((np.ones((N, 1)), phi), axis=1)

        return X_TRAIN, t_TRAIN, phi

    def test_data(self, N_TEST):
        X_TEST = np.random.uniform(0, 1, N_TEST)
        t_TEST = np.sin(2*pi*X_TEST) + E
        g = np.linspace(0, 1, N)
        phi = []
        den = 2 * (s**2)

        for i in range(N_TEST):
            num = (X_TEST[i] - g)**2
            phi.append(np.exp(-1*num/den))

        phi = np.array(phi)
        phi = np.concatenate((np.ones((N_TEST, 1)), phi), axis=1)

        return X_TEST, t_TEST, phi

    def test_err(self, w, X_TEST, t_TEST, lam, phi_test):
        w = np.mean(w, axis=0)
        pred_y = phi_test@w.T
        E = np.mean(pred_y-t_TEST)
        # pdb.set_trace()
        return E

    def weight(self, xtrain, t, N, phi, lam):
        w = []
        f_x = []
        I = np.identity(N+1)
        for i in range(L):
            w_ = (np.linalg.inv(phi[i].T@phi[i]+lam*I))@phi[i].T@t[i]
            fx = phi[i]@w_
            w.append(w_)
            f_x.append(fx)

            # plt.plot(xtrain[i], fx)
        return np.array(w), np.array(f_x)

    def bias_var(self, w, f_x, N, lam):
        f_barx = np.mean(f_x, axis=0)
        h_x = np.sin(2*pi*np.linspace(0, 1, 25))
        bias = np.mean(np.power(f_barx - h_x, 2))

        L_vect = np.mean((f_x-f_barx)**2, axis=0)

        variance = np.mean(L_vect)

        return bias, variance


if __name__ == "__main__":

    Z = Dataset()
    lam = np.linspace(0.01, 15, 300)

    # generating
    X_TEST, t_TEST, phi_test = Z.test_data(1000)
    print(X_TEST.shape)
    print(t_TEST.shape)
    print(phi_test.shape)

    xtrain = []
    ttrain = []
    phi_vect = []

    for _ in range(L):
        X_Train, t_Train, phi = Z.train_data(N)

        xtrain.append(X_Train)
        ttrain.append(t_Train)
        phi_vect.append(phi)

    xtrain = np.array(xtrain)
    ttrain = np.array(ttrain)
    phi_vect = np.array(phi_vect)

    bb = []
    vv = []
    EE = []

    for ite in lam:
        w, f_x = Z.weight(xtrain, ttrain, N, phi_vect, ite)
        b, v = Z.bias_var(w, f_x, N, lam)
        e = Z.test_err(w, X_TEST, t_TEST, lam, phi_test)

        bb.append(b)
        vv.append(v)
        EE.append(e)

    bv = [bb[i] + vv[i] for i in range(len(bb))]
    plt.plot(np.log(lam), bb, label='$bias^2$')
    plt.plot(np.log(lam), vv, label='variance')
    plt.plot(np.log(lam), bv, label='$bias^2$+variance')
    plt.plot(np.log(lam), EE, label='test error')
    plt.xlim(-3, 3)

    plt.legend()
    plt.show()
