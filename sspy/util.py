""" Utility script """

import numpy as np
import warnings

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

def rmse(x,x_):
    """ """
    x = np.atleast_2d(x)
    x_ = np.atleast_2d(x_)
    if x.shape != x_.shape:
        raise ValueError('input shapes don''t match')

    n_x, n_t = x.shape
    serr = (x - x_) ** 2
    return np.sqrt(np.sum(serr,1) / n_t)[:,None]

##
from matplotlib import pyplot as plt

def plot_estimate(x, y, P, c='r', ax = None, colour=None,
                  err_colour=None, alpha = 0.4,
                  linestyle='--', linewidth=3):
    """ """
    if ax is None:
        ax = plt.gca()

    if colour is not None:
        c = colour

    if err_colour is None:
        err_colour = c

    err = 1.96 * np.sqrt(P)
    ymax = y.ravel() + err.ravel()
    ymin = y.ravel() - err.ravel()

    ax.fill_between(x, ymax, ymin, alpha=alpha,
                     facecolor=err_colour, interpolate=True)

    ax.plot(x, y.ravel(), ls=linestyle, lw=linewidth, c=c)

##
import numpy.matlib
def unscented_default_params(n=None):
    """ """
    if n is None:
        n = 0
    return {'alpha': 1., 'beta' : 0., 'kappa': 3. - n}

def unscented_weights(n, params=unscented_default_params()):
    """ Weights for unscented transform """

    if params is None or not isinstance(params,dict):
        params = unscented_default_params(n)
    if 'alpha' not in params or params['alpha'] is None:
        params['alpha'] = unscented_default_params(n)['alpha']
    if 'beta' not in params or params['beta'] is None:
        params['beta'] = unscented_default_params(n)['beta']
    if 'kappa' not in params or params['kappa'] is None:
        params['kappa'] = unscented_default_params(n)['kappa']

    lam = params['alpha'] ** 2 * (n + params['kappa']) - n

    w_m = np.zeros((2*n+1,1))
    w_c = np.zeros((2*n+1,1))

    w_m[0]  = lam / (n + lam)
    w_c[0]  = w_m[0] + (1 - params['alpha'] ** 2 + params['beta'])
    w_m[1:] = np.matlib.repmat((1. - w_m[0])/(2*n), 2*n, 1)
    w_c[1:] = w_m[1:]

    return {'expected': w_m, 'err_cov': w_c}, lam

def unscented_sigmas(x, P, params=unscented_default_params()):
    """ Unscented sigmas """
    n_x = np.atleast_2d(x).shape[0]

    weights, lam = unscented_weights(n_x, params)

    try:
        sP = linalg.cholesky(np.atleast_2d(P * (n_x + lam)))
    except linalg.LinAlgError as lae:
        warnings.warn(str(lae))
        sP = np.sqrt(np.atleast_2d(P * (n_x + lam)))

    sig = np.zeros((n_x, 2*n_x + 1))

    sig[:,1:n_x+1] = sP.T
    sig[:,n_x+1:2*n_x+1] = -sP.T
    sig += np.matlib.repmat(x, 1, 2*n_x + 1)

    return sig, weights

def unscented_transform(x, P, f=None, u=None, params=None):
    """ Unscented transform """
    sig, weights = unscented_sigmas(x, P, params)

    n_x = np.atleast_2d(x).shape[0]
    if f is None:
        f = np.eye(n_x)

    y_ = feval(f, x, u)
    n_y = np.atleast_2d(y_).shape[0]

    #gam = np.zeros((n_y, 2*n_x + 1))

    #for i in range(2*n_x + 1):
    #    gam[:,[i]] = (f,np.atleast_2d(sig[:,[i]]))
    gam = np.hstack([np.atleast_2d(feval(f,np.atleast_2d(s).T,u)) for s in sig.T])

    y  = np.sum(np.matlib.repmat(weights['expected'].T, n_y, 1) * gam,1)[:,None]

    Py  = np.zeros((n_y, n_y))
    Pxy = np.zeros((n_x, n_y))

    for i in range(2*n_x + 1):
        Py += np.matlib.repmat(weights['err_cov'][i],n_y,n_y) * ((gam[:,[i]] - y) @ (gam[:,[i]] - y).T)

        Pxy += np.matlib.repmat(weights['err_cov'][i],n_x,n_y) * ((sig[:,[i]] - x) @ (gam[:,[i]] - y).T)

    return y, Py, Pxy
