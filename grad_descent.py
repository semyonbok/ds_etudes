from typing import Literal

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation

"""
Ideas
 10 Residual plot
 20 Linear model animations
 30 3D mse plot
"""


def update_coef(X, y, w, b, alpha, loss_func: Literal["mse", "rmse"] = "mse"):
    """Assumes X and y are array-likes of same length; w, b, alpha are floats.
    Updates coefficients w and b of a linear model with gradient descent."""

    # find derivatives
    if loss_func == "mse":
        factor = 1
    elif loss_func == "rmse":
        factor = .5 * mse(X, y, w, b)**-.5
    dl_dw = factor * (-2 * X * (y - (w * X + b))).mean()
    dl_db = factor * (-2 * (y - (w * X + b))).mean()

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def train(X, y, w, b, alpha, epochs, report=100, loss_func: Literal["mse", "rmse"] = "mse"):
    """Assumes X and y are array-like of same length; w, b, alpha are float;
    epochs, report are int. Updates linear regression model epochs times"""

    for e in range(epochs):
        w, b = update_coef(X, y, w, b, alpha, loss_func=loss_func)

        # report progress
        if e % report == 0:
            loss = eval(loss_func)(X, y, w, b)
            print(
                f"epoch: {e:.4f}, w: {w:.4f}, b: {b:.4f}, {loss_func}: {loss:.4}")

    return w, b


def mse(X, y, w, b):
    """Assumes X and y are array-like of same length; w, b, alpha are float.
    Computes mean squared error."""

    loss = ((y - (w * X + b))**2).mean()

    return loss


def rmse(X, y, w, b):
    """Assumes X and y are array-like of same length; w, b, alpha are float.
    Computes root mean squared error."""

    loss = ((y - (w * X + b))**2).mean()**.5

    return loss
