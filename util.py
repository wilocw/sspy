""" Utility script """

import numpy as np

def feval(f, x, params=None):
    """
    Function evaluation

    Takes function handle, lambda function or matrix and applies to input x
    """
    if callable(f):
        # f is a callable function
        if params is None:
            try:
                y = f(x)
            except ValueError as ve:
                print(ve)
                print("Unable to evaluate f(x)")
                y = x
        else:
            try:
                y = f(x, params)
            except ValueError as ve:
                print(ve)
                print("Unable to evaluate f(x,p)")
                y = x
    elif isinstance(f, np.ndarray):
        # f is nd array
        if params is None:
            try:
                y = f @ x
            except ValueError as ve:
                print(ve)
                print("Unable to evaluate f * x")
                y = x
        else:
            try:
                y = f @ x + params
            except ValueError as ve:
                print(ve)
                print("Unable to evaluate f * x + p")
                y = x
    else:
        raise ValueError('Unable to infer type of f to evaulate')

    return y

def model_noiseless(x0, f, uf = None, h=None, uh=None, n:int=500):
    """
    Simulate a noiseless system
        x_t = f(x_{t-1}, uf_t)
        y_t = h(x_t, uh_t)
    """
    n_x = np.atleast_2d(x0).shape[0]

    x = np.zeros((n_x, n))

    x[:,0] = x0.ravel()

    if h is not None:
        n_y = np.atleast_2d(feval(h, x0,  uh)).shape[0]
        y = np.zeros((n_y, n))
        y[:,0] = feval(h, x[:,0][:,None], uh).ravel()
    else:
        y = None

    for i_ in range(1,n):
        x[:,i_] = feval(f, x[:,i_-1][:,None], uf).ravel()
        if h is not None:
            y[:,i_] = feval(h, x[:,i_][:,None], uh).ravel()

    return x, y

##
from numpy.random import randn
from scipy import linalg
def model_noisy(x0, f, Q, uf = None, h=None, R=None, uh=None, n:int=500):
    """
    Simulate a noisy system
        x_t = f(x_{t-1}, uf_t; N(0, Q))
        y_t = h(x_t, uh_t; N(0, R))
    """
    n_x = np.atleast_2d(x0).shape[0]

    x = np.zeros((n_x, n))
    try:
        sQ = linalg.cholesky(np.atleast_2d(Q))
    except linalg.LinAlgError:
        sQ = np.sqrt(np.atleast_2d(Q))

    x[:,0] = (x0 + sQ @ np.random.randn(n_x, 1)).ravel()

    if h is not None:
        n_y = np.atleast_2d(feval(h, x0, uh)).shape[0]
        # True observation state
        y_true = np.zeros((n_y, n))
        y_true[:,0] = feval(h, x[:,0][:,None], uh).ravel()
        # Noisy measurements
        y_noisy = np.zeros_like(y_true)
        try:
            sR = linalg.cholesky(np.atleast_2d(R))
        except linalg.LinAlgError:
            sR = np.sqrt(np.atleast_2d(R))
        y_noisy[:,0] = (y_true[:,0][:,None] + sR @ randn(n_y, 1)).ravel()
    else:
        y_true  = None
        y_noisy = None

    for i_ in range(1, n):
        x[:,i_] = (feval(f, x[:,i_-1][:,None], uf) + sQ @ np.random.randn(n_x, 1)).ravel()
        y_true[:,i_]  = feval(h, x[:,i_][:,None], uh)
        y_noisy[:,i_] = (y_true[:,i_][:,None] + sR @ randn(n_y, 1)).ravel()

    return x, y_true, y_noisy

##
from matplotlib import pyplot as plt

def plot_estimate(x, y, P, c='r', colour=None,
                  err_colour=None, alpha = 0.4,
                  linestyle='--', linewidth=3):
    """ """
    if colour is not None:
        c = colour

    if err_colour is None:
        err_colour = c

    err = 1.96 * np.sqrt(P)
    ymax = y.ravel() + err.ravel()
    ymin = y.ravel() - err.ravel()

    plt.fill_between(x, ymax, ymin, alpha=alpha,
                     facecolor=err_colour, interpolate=True)

    plt.plot(x, y.ravel(), ls=linestyle, lw=linewidth, c=c)
