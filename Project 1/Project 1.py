import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read From DATA
DATA = pd.read_excel('proj1DATAset.xlsx')

# Fill Null Values with Mean
DATA.fillna(DATA['Horsepower'].mean(), inplace=True)

# Turn pd DATAfram into numpy array
#predictor = weight
#target = horsepower
x_predictor = np.array(DATA['Weight'])
t = np.array(DATA['Horsepower'])

# N number of samples in DATAset
# D dimension
N = DATA.shape[0]
D = 1

# Design Matrix X
X = np.append(x_predictor.reshape(
    N, 1), np.ones(N).reshape(N, 1), axis=1)

X_AXIS = np.linspace(1500, 5000, 500)

# ---Closed Form Solution---


def closed_form():
    '''
    Returns the plot and weight vector through closed form solution
    '''
    weight = np.linalg.inv(X.T@X)@X.T@(t)
    print(f"Closed Form Weight = {weight}")
    closed_form = weight[0] * X_AXIS + weight[1]
    plt.subplot(1, 2, 1)
    plt.scatter(x_predictor, t, c='red', marker='x', lw=0.9)
    plt.plot(X_AXIS, closed_form, lw=2, label='Closed Form')
    plt.title('Closed Form Solution')
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.legend()

# ---Gradient Descent Iterative---


def iterative_solution():
    '''
    Returns the plot and weight vector through iterative gradient descent solution
    '''
    # Normalization
    x_predictor_normal = x_predictor / x_predictor.max()
    w = np.random.normal(size=2)
    learning_rate = 1e-3
    print(w)
    # Design Matrix after normalization
    X = np.append(x_predictor_normal.reshape(
        N, 1), np.ones(N).reshape(N, 1), axis=1)

    # For 1000 epochs
    for _ in range(1000):
        loss_func = 2 * (w.T@(X.T@X)) - 2 * t.T@X
        w = w - learning_rate * loss_func
        y = X@np.array(w).T
        err = np.mean((y - t)**2)
        print(err)

    # Denormalization of weight vector
    w[0] = w[0] / x_predictor.max()
    print(f"Gradient Descent Weight = {w}")
    plt.subplot(1, 2, 2)
    ite_soln = w[0] * X_AXIS + w[1]
    y = X@w.T
    print(w.shape)
    print(X_AXIS.shape, ite_soln.shape)

    plt.scatter(x_predictor, t, marker='x', lw=0.9, c='red')
    plt.plot(X_AXIS, ite_soln, lw=2, color='black', label='Gradient Descent')
    plt.title('Gradient Descent')
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.legend()


if __name__ == "__main__":
    closed_form()
    iterative_solution()

plt.show()
