""" Module for smooth classes """

import numpy as np
from scipy import linalg
import abc
from filters import _Filter

class _Smoother(metaclass=abc.ABCMeta):
    """
    Abstract class for smoothers for smoothing filter
    estimates in a state space model

    dx(t)/dt = f(x(t), u(t), xi(t))
    y(t) = h(x(t), u(t), nu(t))

    x  : latent state
    y  : observable state
    u  : input/control
    xi : process noise
    nu : observation noise
    ===
    """
    @abc.abstractmethod
    def smooth(self):
        """ TODO: write docstring """
        NotImplemented

class RauchTungStriebelSmoother(_Smoother):
    """ """

    def __init__(self, states,
                 F = None, Q = None, Uf = None,
                 _verbose:bool = False):
        """ """
        self.filtered = states
        self.smoothed = [{}] * len(self.filtered)
        self.process = {'f': np.array(F),
                        'U': np.array(Uf),
                        'Q': np.array(Q)}
        self._verbose = _verbose
        self._tx = len(self.filtered) - 1
        self.smoothed[self._tx] = self.filtered[self._tx]
        self._tx -= 1

    @classmethod
    def from_filter(cls, filter:_Filter):
        " Initialise RTS smoother from filter history "
        states = filter._history['updates']
        F = filter.process['f']
        Q = filter.process['Q']
        U = filter.process['U']
        return cls(states, F, Q, U)

    def smooth(self):
        """ """
        while self._tx > -1:
            self.smooth_incremental()

        return self.smoothed


    def smooth_incremental(self, Q = None, u = None):
        """ """
        if Q is None:
            Q = self.process['Q']
        else:
            self.process['Q'] = Q

        x_  = self.filtered[self._tx]['expected']

        if u is None:
            x_p = self.process['f'] @ x_
        else:
            x_p = self.process['f'] @ x_ + self.process['U'] @ u

        P_  = self.filtered[self._tx]['err_cov']
        P_p = self.process['f'] @ P_ @ self.process['f'].T + Q

        S = P_ @ self.process['f'] @ linalg.inv(P_p)

        x_t = self.smoothed[self._tx+1]['expected']
        P_t = self.smoothed[self._tx+1]['err_cov']

        x = x_ + S @ (x_t - x_p)
        P = P_ + S @ (P_t - P_p) @ S.T

        self.smoothed[self._tx] = {'expected': x, 'err_cov': P}
        self._tx -= 1


class KalmanSmoother(RauchTungStriebelSmoother):
    """ Alias class for Rauch-Tung-Striebel Smoother """
