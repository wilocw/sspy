""" Module for filter classes """

import numpy as np
from scipy import linalg
import abc

from util import feval

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
    """
    Linear Discrete Kalman Filter

    TODO: write docstring
    """

    def __init__(self,
                 x0, P0,
                 F = None, Q0 = None,
                 H = None, R0 = None,
                 Uf = None, Uh = None,
                 _verbose:bool= False):
        """ TODO: write docstring """
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
        """ TODO: write docstring """
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
        """ TODO: write docstring """

        x_ = self.state['expected']
        P_ = self.state['err_cov']

        if u is None:
            y_ = self.observe['h'] @ x_
        else:
            y_ = self.observe['h'] @ x_ + self.observe['U'] @ u

        if R is None:
            R = self.observe['R']
        else:
            self.observe['R'] = R

        P_xy = P_ @ self.observe['h'].T
        P_y  = self.observe['h'] @ P_xy + R

        K = P_xy @ linalg.inv(P_y)

        x = x_ + K @ np.atleast_2d(y - y_)
        self.state['expected'] = x.copy()

        P = P_ - K @ P_y @ K.T
        self.state['err_cov'] = P.copy()

        self._history['updates'].append(self.state.copy())


class LinearDiscreteKalmanFilter(KalmanFilter):
    """ Alias class for Kalman Filter """

### === Non-Linear Kalman Filters ===
class ExtendedKalmanFilter(KalmanFilter):
    """
    First Order Extended Kalman Filter

    TODO: write docstring
    """

    def __init__(self,
                 x0, P0,
                 f  = None, Q0 = None,
                 Jf = None, JQ = None, # Jacobians
                 h  = None, R0 = None,
                 Jh = None, JR = None, # Jacobians
                 _verbose:bool= False):
        """ TODO: write docstring """
        self.state   = {'expected': np.atleast_2d(x0),
                        'err_cov' : np.atleast_2d(P0)}
        if f is None:
            f = np.eye(x0.shape[0])
        if Q0 is None:
            Q0 = np.zeros((x0.shape[0], x0.shape[0]))
        if Jf is None:
            Jf = np.eye(x0.shape[0])

        self.process = {'f'         : f,
                        'Q'         : Q0,
                        'Jacobian_x': Jf,
                        'Jacobian_Q': JQ}

        if h is None:
            h = np.eye(x0.shape[0])
        if R0 is None:
            y_ = feval(h,x0)
            R0 = np.zeros((y_.shape[0], y_.shape[0]))
        if Jh is None:
            Jh = np.eye(x0.shape[0])

        self.observe = {'h'         : h,
                        'R'         : R0,
                        'Jacobian_x': Jh,
                        'Jacobian_R': JQ}

        self._history = {'predictions': [],
                         'updates': []}
        self._history['updates'].append(self.state)
        self.initial = self.state.copy()
        self._verbose = _verbose

    def predict(self, Q = None, u = None):
        """ TODO: write docstring """

        x_ = self.state['expected']
        P_ = self.state['err_cov']

        if u is None:
            x  = feval(self.process['f'], x_)
            F_ = feval(self.process['Jacobian_x'], x_)
            if self.process['Jacobian_Q'] is None:
                w_ = np.eye(x_.shape[0])
            else:
                w_ = feval(self.process['Jacobian_Q'], x_)
        else:
            x  = feval(self.process['f'], x_, u)
            F_ = feval(self.process['Jacobian_x'], x_, u)
            if self.process['Jacobian_Q'] is None:
                w_ = np.eye(x_.shape[0])
            else:
                w_ = feval(self.process['Jacobian_Q'], x_, u)
        self.state['expected'] = x.copy()

        if Q is None:
            Q = self.process['Q']
        else:
            self.process['Q'] = Q

        P = F_ @ P_ @ F_.T + w_ @ Q @ w_.T
        self.state['err_cov'] = P.copy()
        if self._verbose:
            print(self.state)
        self._history['predictions'].append(self.state.copy())

    def update(self, y, R = None, u = None):
        """ TODO: write docstring """

        x_ = self.state['expected']
        P_ = self.state['err_cov']

        if self._verbose:
            print(x_)

        if u is None:
            y_ = np.atleast_2d(feval(self.observe['h'], x_))
            if self._verbose:
                print(y_)
            H_ = feval(self.observe['Jacobian_x'], x_)
            if self.observe['Jacobian_R'] is None:
                v_ = np.eye(y_.shape[0])
            else:
                v_ = feval(self.observe['Jacobian_R'], x_)
        else:
            y_ = feval(self.observe['h'], x_, u)
            H_ = feval(self.observe['Jacobian_x'], x_, u)
            if self.observe['Jacobian_R'] is None:
                v_ = np.eye(y_.shape[0])
            else:
                v_ = feval(self.observe['Jacobian_R'], x_, u)

        if R is None:
            R = self.observe['R']
        else:
            self.observe['R'] = R

        P_xy = P_ @ H_.T
        P_y  = H_ @ P_xy + v_ @ R @ v_.T

        K = P_xy @ linalg.inv(P_y)

        # The reshape part fixes some weird scalar issues
        x = x_ + K @ np.atleast_2d(y - y_)

        self.state['expected'] = x.copy()
        if self._verbose:
            print('x is ' + str(x))

        P = P_ - K @ P_y @ K.T
        self.state['err_cov'] = P.copy()

        self._history['updates'].append(self.state.copy())


class FirstOrderExtendedKalmanFilter(ExtendedKalmanFilter):
    """ Alias class for Extended Kalman Filter """
    pass

class SecondOrderExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Second Order Extended Kalman Filter
    TODO: write class
    """
    NotImplemented
