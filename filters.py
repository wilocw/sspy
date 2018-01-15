""" Module for filter classes """

import numpy as np
from scipy import linalg
import abc

class _Filter(metaclass=abc.ABCMeta):
    """
    Abstract class for filters for predicting and updating
    estimates in a state space model

    dx(t)/dt = f(x(t), u(t), xi(t))
    y(t) = h(x(t),u(t),nu(t))

    x  : latent state
    y  : observable state
    u  : input/control
    xi : process noise
    nu : observation noise
    ===
    """
    @abc.abstractmethod
    def predict(self):
        """ TODO: write docstring """
        NotImplemented

    @abc.abstractmethod
    def update(self):
        """ TODO: write docstring """
        NotImplemented

class KalmanFilter(_Filter):
    """ """

    def __init__(self,
                 x0, P0,
                 F = None, Q0 = None,
                 H = None, R0 = None,
                 Uf = None, Uh = None,
                 _verbose:bool= False):
        """ """
        self.state   = {'expected': np.array(x0),
                        'err_cov' : np.array(P0)}
        if F is None:
            F = np.eye(x0.shape[0])
        if Q0 is None:
            Q0 = np.zeros_like(F)
        self.process = {'f': np.array(F),
                        'U': np.array(Uf),
                        'Q': np.array(Q0)}

        if H is None:
            H = np.eye(x0.shape[0])
        if R0 is None:
            R0 = np.zeros((H.shape[0], H.shape[0]))
        self.observe = {'h': np.array(H),
                        'U': np.array(Uh),
                        'R': np.array(R0)}
        self._history = {'predictions': [],
                         'updates': []}
        self._history['updates'].append(self.state)
        self.initial = self.state.copy()
        self._verbose = _verbose

    def predict(self, Q = None, u=None):
        """ """
        x_ = self.state['expected']
        P_ = self.state['err_cov']
        if u is None:
            x = self.process['f'] @ x_
        else:
            x = self.process['f'] @ x_ + self.process['U'] @ u
        self.state['expected'] = x.copy()

        if Q is None:
            Q = self.process['Q']
        else:
            self.process['Q'] = Q

        P = self.process['f'] @ P_ @ self.process['f'].T + Q
        self.state['err_cov'] = P.copy()
        if self._verbose:
            print(self.state)
        self._history['predictions'].append(self.state.copy())

    def update(self, y, R = None, u = None):
        """ """
        if R is None:
            R = self.observe['R']
        else:
            self.observe['R'] = R

        P_xy = self.state['err_cov'] @ self.observe['h'].T
        P_y  = self.observe['h'] @ P_xy + R

        K = P_xy @ linalg.inv(P_y)

        x_ = self.state['expected']
        P_ = self.state['err_cov']

        if u is None:
            y_ = self.observe['h'] @ x_
        else:
            y_ = self.observe['h'] @ x_ + self.observe['U'] @ u

        x = x_ + K @ (y - y_)
        self.state['expected'] = x.copy()

        P = P_ - K @ P_y @ K.T
        self.state['err_cov'] = P.copy()

        self._history['updates'].append(self.state.copy())


class LinearDiscreteKalmanFilter(KalmanFilter):
    """ Alias class for Kalman Filter """
