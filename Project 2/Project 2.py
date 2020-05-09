import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import exp, sqrt, pi, sin
# random seed for consistent output
# np.random.seed(1)
E = np.random.normal(0, 0.3)
M = range(0, 10)
PLOTNUM = 1


def gen_train(N_TRAIN):
    # ---Generate Training Set---
    X_TRAIN = np.random.uniform(0, 1, N_TRAIN)
    t_TRAIN = []
    t_TRAIN = np.sin(2*pi*X_TRAIN)+E

    # ---Generate Testing Set---
    N_TEST = 100
    X_TEST = np.random.uniform(0, 1, N_TEST)
    t_TEST = []
    t_TEST = np.sin(2*pi*X_TEST)+E

    return X_TRAIN, t_TRAIN, X_TEST, t_TEST, N_TRAIN


def get_weight(M, N, X, t):
    # ---Generate Weight From the Given Train Sets---
    phi = np.ones(M+1)
    for i in range(N):
        temp = []
        temp = np.ones(1)
        for j in range(M):
            temp = np.append(temp, X[i]**(j+1))
        phi = np.vstack([phi, temp])
    phi = np.delete(phi, 0, 0)

    weight = (np.linalg.inv(phi.T@phi))@phi.T@t
    return weight, phi


def get_error(M, N, X, t):
    # ---Get Error ---
    weight, phi = get_weight(M, N, X, t)
    LS = weight.T@phi.T@phi@weight - 2*(t.T@phi@weight)+t.T@t
    E_RMS = sqrt(LS/N)
    return E_RMS


def plot_graph(X_TRAIN, t_TRAIN, X_TEST, t_TEST, PLOTNUM):
    train_err = []
    test_err = []
    for ite in M:
        # appends train and test error for each non-linear model of degree M
        train_err.append(get_error(ite, N_TRAIN, X_TRAIN, t_TRAIN))
        test_err.append(get_error(ite, 100, X_TEST, t_TEST))

    # plot of the graphs
    plt.subplot(1, 2, PLOTNUM)

    plt.xticks(np.arange(0, 10, step=1))
    sns.lineplot(M, train_err, label="Training", marker="o")
    sns.lineplot(M, test_err, label="Testing", marker="o", color="red")
    plt.xlabel("M")
    plt.ylabel("E_rms")
    plt.title(f"Number of training data: {N_TRAIN}")


if __name__ == "__main__":
    X_TRAIN, t_TRAIN, X_TEST, t_TEST, N_TRAIN = gen_train(10)
    plot_graph(X_TRAIN, t_TRAIN, X_TEST, t_TEST, PLOTNUM)
    PLOTNUM += 1

    X_TRAIN, t_TRAIN, X_TEST, t_TEST, N_TRAIN = gen_train(100)
    plot_graph(X_TRAIN, t_TRAIN, X_TEST, t_TEST, PLOTNUM)

    plt.show()
