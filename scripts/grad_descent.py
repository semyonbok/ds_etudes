
from typing import Literal, Union

import logging
import numpy as np

logging.basicConfig(
    filename="gd_log.log",
    level=logging.INFO,
    format='%(asctime)s %(message)s'
    )

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


def update_coef(
        X, y, w, b, alpha,
        cost_func: Literal["mse", "rmse"] = "mse",
        regularize=False, **reg_kwargs
        ):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `cost_func` is string representing a cost
    function; `regularize` is boolean; `reg_kwargs` are arguments to pass to
    regularization function: `C` and `power`.
    Updates coefficients `w` and `b` of a linear model with gradient descent.
    """

    # find derivatives
    if cost_func == "mse":
        factor = 1
    elif cost_func == "rmse":
        factor = .5 * mse(X, y, w, b)**-.5
    error = np.dot(X, w) + b - y  # abs error per each data entry
    if X.ndim > 1:
        dl_dw = factor * (X * error.reshape((-1, 1))).mean(axis=0)
    else:
        dl_dw = factor * (X * error).mean()
    dl_db = factor * error.mean()

    # add regulariztion gradient term
    dl_dw += reg_grad(regularize, X, w, **reg_kwargs)

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def log_update_coef(
        X, y, w, b, alpha,
        border_func=lambda X, w, b: np.dot(X, w) + b,
        regularize=False, **reg_kwargs
        ):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `border_func` is a decision border
    function; `regularize` is boolean; `reg_kwargs` are arguments to pass to
    regularization function: `C` and `power`.
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

    # add regulariztion gradient term
    dl_dw += reg_grad(regularize, X, w, **reg_kwargs)

    # update coef
    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def train(
        X, y, w, b, alpha, epochs,
        report=100, cost_func: Literal["mse", "rmse"] = "mse",
        regularize=False, **reg_kwargs
        ):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `epochs`, `report` are ints; `cost_func` is
    string representing a cost function; `regularize` is boolean; `reg_kwargs`
    are arguments to pass to regularization function: `C` and `power`.
    Updates linear regression model `epochs` times."""
    validate_input(X, y, w)
    for e in range(epochs):
        w, b = update_coef(
            X, y, w, b, alpha,
            cost_func=cost_func,
            regularize=regularize, **reg_kwargs
            )

        # report progress
        if e % report == 0:
            cost = eval(cost_func)(X, y, w, b)
            print(
                f"epoch: {e}, w: {w}, b: {b:.4f}, {cost_func}: {cost:.4f}")

    return w, b


def log_train(
        X, y, w, b, alpha, epochs,
        report=100, border_func=lambda X, w, b: np.dot(X, w) + b,
        regularize=False, **reg_kwargs
        ):
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `epochs`, `report` are ints; `border_func`
    is a decision border function; `regularize` is boolean; `reg_kwargs` are
    arguments to pass to regularization function: `C` and `power`.
    Updates logistic regression model `epochs` times."""
    validate_input(X, y, w)
    for e in range(epochs):
        w, b = log_update_coef(
            X, y, w, b, alpha,
            border_func=border_func,
            regularize=regularize, **reg_kwargs
            )

        # report progress
        if e % report == 0:
            cost = log_cost(X, y, w, b, border_func)
            logging.info(
                f"epoch: {e}, w: {w}, b: {b:.4f}, cost: {cost:.4f}"
                )
            print(
                f"epoch: {e}, w: {w}, b: {b:.4f}, cost: {cost:.4f}"
                )

    return w, b


def mse(X, y, w, b, regularize=False, **reg_kwargs) -> float:
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b` is float; `regularize` is boolean; `reg_kwargs` are arguments to
    pass to regularization function: `C` and `power`.
    Computes mean squared error."""
    reg_ = reg(regularize, X, w, **reg_kwargs)
    return reg_ + (((np.dot(X, w) + b) - y)**2).mean()


def rmse(X, y, w, b, regularize=False, **reg_kwargs) -> float:
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b` is float; `regularize` is boolean; `reg_kwargs` are arguments to
    pass to regularization function: `C` and `power`.
    Computes root mean squared error."""
    reg_ = reg(regularize, X, w, **reg_kwargs)
    return reg_ + (((np.dot(X, w) + b) - y)**2).mean()**.5


def log_cost(X, y, w, b, border_func=lambda X, w, b: np.dot(X, w) + b,
             regularize=False, **reg_kwargs) -> float:
    """Assumes `X` and `y` are array-like of same length; `w` is float when `X`
    is 1-D array and 1-D array of same length as `X`'s width if `X` is 2-D
    array; `b`, `alpha` are floats; `border_func` is a decision border
    function; `regularize` is boolean; `reg_kwargs` are arguments to pass to
    regularization function: `C` and `power`.
    Computes log-likelihood error."""
    reg_ = reg(regularize, X, w, **reg_kwargs)
    z = border_func(X, w, b)
    f = 1 / (1 + np.exp(-z))
    return reg_ + (-y * np.log(f) - (1 - y) * np.log(1 - f)).mean()


def reg(regularize, X, w, C=1e-3, power=2) -> float:
    """Assumes `regularize` is boolean; `X` is array-like, `w` is either
    array-like or float; `C` is either float or array-like of the same length
    as `w`; `power` is int.
    Computes regulariztion term for a regression cost function."""
    if regularize:
        return np.sum(np.abs(w)**power * C) / X.shape[0]
    return 0.0


def reg_grad(regularize, X, w, C=1e-3, power=2) -> Union[float, np.ndarray]:
    """Assumes `regularize` is boolean; `X` is array-like, `w` is either
    array-like or float; `C` is either float or array-like of the same length
    as `w`; `power` is int.
    Computes gradient of regulariztion term for a regression cost function."""
    if regularize:
        return power * (np.abs(w)**(power - 1) * C) / X.shape[0]
    return 0.0


def validate_input(X, y, w):
    """Assumes `X` and `y` are array-like, `w` is either array-like or scalar.
    Asserts some conditions for regression model training."""
    assert X.shape[0] == y.shape[0], (
        "\nNumber of data entries and target "
        f"values are not equal.\nThere are {X.shape[0]} data entries and "
        f"{y.shape[0]} target values."
        )

    if X.ndim == 1:
        assert np.isscalar(w), (
            "\n`w` must be scalar when data "
            f"entries have only one feature. `w` passed: {w}"
        )
    else:
        assert not np.isscalar(w), (
            "\n`w` must be array-like when "
            f"data entries have more than one feature. `w` passed: {w}"
            )
        assert X.shape[1] == w.shape[0], (
            "\nNumber of features in data entries "
            "and number of weight coefficients are not equal.\nThere are "
            f"{X.shape[1]} features and {w.shape[0]} weight coefficients."
            )
