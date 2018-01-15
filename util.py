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
