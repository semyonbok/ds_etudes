
from typing import Literal
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

"""
Ideas
 10 Residual plot
 11 Linear model animations
 12 3D mse and/or contour plot
 13 Learning curves for multiple learning rates
 20 Try varying learning rate for each features
 21 Add regularization
"""


def update_coef(X, y, w, b, alpha, cost_func: Literal["mse", "rmse"] = "mse"):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `cost_func` is string representing a cost function.
    Updates coefficients `w` and `b` of a linear model with gradient descent."""

    # find derivatives
    if cost_func == "mse":
        factor = 1
    elif cost_func == "rmse":
        factor = .5 * mse(X, y, w, b)**-.5

    # if isinstance(w, Number):
    #     dl_dw = factor * (2 * X * ((w * X + b) - y)).mean()
    #     dl_db = factor * (2 * ((w * X + b) - y)).mean()
    # else:
    error = np.dot(X, w) + b - y  # abs error per each data entry
    if X.ndim > 1:
        dl_dw = factor * (X * error.reshape((-1, 1))).mean(axis=0)
    else:
        dl_dw = factor * (X * error).mean()
    dl_db = factor * error.mean()

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def log_update_coef(X, y, w, b, alpha, border_func=lambda X, w, b: np.dot(X, w) + b):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `border_func` is a decision border function.
    Updates coefficients `w` and `b` of a logistic regression model with
    gradient descent."""

    # find derivatives
    z = border_func(X, w, b)
    error = (1 / (1 + np.exp(-z))) - y
    if X.ndim > 1:
        dl_dw = (X * error.reshape((-1, 1))).mean(axis=0)
    else:
        dl_dw = (X * error).mean()
    dl_db = error.mean()

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def train(X, y, w, b, alpha, epochs,
          report=100, cost_func: Literal["mse", "rmse"] = "mse"):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `epochs`, `report` are ints; `cost_func` is
    string representing a cost function.
    Updates linear regression model `epochs` times."""

    for e in range(epochs):
        w, b = update_coef(X, y, w, b, alpha, cost_func=cost_func)

        # report progress
        if e % report == 0:
            cost = eval(cost_func)(X, y, w, b)
            print(
                f"epoch: {e}, w: {w}, b: {b:.4f}, {cost_func}: {cost:.4f}")

    return w, b


def log_train(X, y, w, b, alpha, epochs,
              report=100, border_func=lambda X, w, b: np.dot(X, w) + b):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `epochs`, `report` are ints; `border_func`
    is a decision border function.
    Updates logistic regression model `epochs` times."""

    for e in range(epochs):
        w, b = log_update_coef(X, y, w, b, alpha, border_func=border_func)

        # report progress
        if e % report == 0:
            cost = log_cost(X, y, w, b, border_func)
            print(
                f"epoch: {e}, w: {w}, b: {b:.4f}, cost: {cost:.4f}")

    return w, b


def mse(X, y, w, b) -> float:
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; b is float.
    Computes mean squared error."""
    return (((np.dot(X, w) + b) - y)**2).mean()


def rmse(X, y, w, b) -> float:
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; b is float.
    Computes root mean squared error."""
    return (((np.dot(X, w) + b) - y)**2).mean()**.5


def log_cost(X, y, w, b, border_func=lambda X, w, b: np.dot(X, w) + b):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `border_func` is a decision border function.
    Computes log-likelihood error.
    """
    z = border_func(X, w, b)
    f = 1 / (1 + np.exp(-z))
    return (-y * np.log(f) - (1 - y) * np.log(1 - f)).mean()
