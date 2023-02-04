from numbers import Number
from typing import Literal
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

"""
Ideas
 10 Residual plot
 20 Linear model animations
 30 3D mse plot
"""


def update_coef(X, y, w, b, alpha, loss_func: Literal["mse", "rmse"] = "mse"):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `loss_func` is string representing a loss function.
    Updates coefficients `w` and `b` of a linear model with gradient descent."""

    # find derivatives
    if loss_func == "mse":
        factor = 1
    elif loss_func == "rmse":
        factor = .5 * mse(X, y, w, b)**-.5
    
    if isinstance(w, Number):
        dl_dw = factor * (2 * X * ((w * X + b) - y)).mean()
        dl_db = factor * (2 * ((w * X + b) - y)).mean()
    else:
        error = np.dot(X, w) + b - y  # abs error per each data entry
        dl_dw = factor * (X * error.reshape((-1,1))).mean(axis=0)
        dl_db = factor * error.mean()

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def train(X, y, w, b, alpha, epochs,
    report=100, loss_func: Literal["mse", "rmse"] = "mse"):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `epochs`, `report` are ints; `loss_func` is
    string representing a loss function.
    Updates linear regression model `epochs` times."""

    for e in range(epochs):
        w, b = update_coef(X, y, w, b, alpha, loss_func=loss_func)

        # report progress
        if e % report == 0:
            loss = eval(loss_func)(X, y, w, b)
            print(
                f"epoch: {e}, w: {w}, b: {b:.4f}, {loss_func}: {loss:.4f}")

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
