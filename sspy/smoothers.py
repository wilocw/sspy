""" Module for smooth classes """

import numpy as np
from scipy import linalg
import abc

from sspy.filters import _Filter
from sspy.util import feval, unscented_transform, unscented_default_params

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
    """ TODO: write docstring """

    def __init__(self, states,
                 F = None, Q = None, Uf = None,
                 _verbose:bool = False):
        """ """
        self.filtered = states
        self.smoothed = [{}] * len(self.filtered)
        self.process = {'f': F,
                        'U': Uf,
                        'Q': Q}
        self._verbose = _verbose
        self._tx = len(self.filtered) - 1
        self.smoothed[self._tx] = self.filtered[self._tx]
        self._tx -= 1

    @classmethod
    def from_filter(cls, filter:_Filter):
        """ Initialise RTS smoother from filter history """
        states = filter._history['updates']

        F = filter.process['f']
        Q = filter.process['Q']
        U = filter.process['U']

        return cls(states, F, Q, U)

    def smooth(self):
        """ TODO: write docstring """
        while self._tx > -1:
            self.smooth_incremental()

        return self.smoothed


    def smooth_incremental(self, Q = None, u = None):
        """ TODO: write docstring """
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

class ExtendedRauchTungStriebelSmoother(RauchTungStriebelSmoother):
    """
    First Order Extended Rauch-Tung-Striebel Smoother

    TODO: write docstring
    """

    def __init__(self, states,
                 f = None, Q = None,
                 Jf = None, JQ = None, # Jacobians
                 _verbose:bool = False):
        """ """
        # self.filtered = states
        # self.smoothed = [{}] * len(self.filtered)
        # self.process = {'f'         : f,
        #                 'Q'         : Q,
        #                 'Jacobian_x': Jf,
        #                 'Jacobian_Q': JQ}
        # self._verbose = _verbose
        # self._tx = len(self.filtered) - 1
        # self.smoothed[self._tx] = self.filtered[self._tx]
        # self._tx -= 1
        super().__init__(states, f, Q, _verbose)
        self.process['Jacobian_x'] = Jf
        self.process['Jacobian_Q'] = JQ

    @classmethod
    def from_filter(cls, filter:_Filter):
        """ Initialise ERTS smoother from filter history """
        states = filter._history['updates']

        f  = filter.process['f']
        Q  = filter.process['Q']
        Jx = filter.process['Jacobian_x']
        JQ = filter.process['Jacobian_Q']

        return cls(states, f, Q, Jx, JQ)

    def smooth_incremental(self, Q = None, u = None):
        """ TODO: write docstring """
        if Q is None:
            Q = self.process['Q']
        else:
            self.process['Q'] = Q

        x_ = self.filtered[self._tx]['expected']

        # if u is None:
        #     x_p = feval(self.process['f'], x_)
        #     F_ = feval(self.process['Jacobian_x'], x_)
        #     if self.process['Jacobian_Q'] is None:
        #         w_ = np.eye(x_.shape[0])
        #     else:
        #         w_ = feval(self.process['Jacobian_Q'], x_)
        # else:
        x_p = feval(self.process['f'], x_, u)
        F_ = feval(self.process['Jacobian_x'], x_, u)
        if self.process['Jacobian_Q'] is None:
            w_ = np.eye(x_.shape[0])
        else:
            w_ = feval(self.process['Jacobian_Q'], x_, u)

        P_  = self.filtered[self._tx]['err_cov']
        P_p = F_ @ P_ @ F_.T + w_ @ Q @ w_.T

        S = P_ @ F_ @ linalg.inv(P_p)

        x_t = self.smoothed[self._tx+1]['expected']
        P_t = self.smoothed[self._tx+1]['err_cov']

        x = x_ + S @ np.atleast_2d(x_t - x_p)
        P = P_ + S @ np.atleast_2d(P_t - P_p) @ S.T

        self.smoothed[self._tx] = {'expected': x, 'err_cov': P}
        self._tx -= 1

class UnscentedRauchTungStriebelSmoother(RauchTungStriebelSmoother):
    """
    Rauch-Tung-Striebel Smoother that uses the Unscented Transform

    TODO: write docstring
    """

    def __init__(self, states,
                 f = None, Q = None,
                 params = unscented_default_params(),
                 augment:bool = False,
                 _verbose:bool = False):
        """ TODO: write docstring """
        super().__init__(states, f, Q, _verbose)
        self.params = params
        self.augment = augment


    @classmethod
    def from_filter(cls, filter:_Filter):
        """ Initialise URTS smoother from filter history """
        states = filter._history['updates']

        f = filter.process['f']
        Q = filter.process['Q']
        params = filter.params
        augment = filter.augment

        return cls(states, f, Q, params, augment)

    def smooth_incremental(self, Q = None, u = None):
        """ TODO: write docstring """

        if Q is None:
            Q = self.process['Q']
        else:
            self.process['Q'] = Q

        x_ = self.filtered[self._tx]['expected']
        P_ = self.filtered[self._tx]['err_cov']

        if self.augment:
            x_ = np.vstack([x_,np.zeros_like(x_)])
            P_ = np.hstack([np.vstack([P_, np.zeros_like(P_)]),
                            np.vstack([np.zeros_like(P_), Q])])

        x_p, P_p, P_x = unscented_transform(x_, P_, self.process['f'], u, self.params)

        if not self.augment:
           P_p += Q

        S = P_x @ linalg.inv(P_p)

        x_t = self.smoothed[self._tx+1]['expected']
        P_t = self.smoothed[self._tx+1]['err_cov']

        x = x_ + S @ (x_t - x_p)
        P = P_ + S @ (P_t - P_p) @ S.T

        self.smoothed[self._tx] = {'expected': x, 'err_cov': P}
        self._tx -= 1

### === Alias classes ===
class KalmanSmoother(RauchTungStriebelSmoother):
    """ Alias class for Rauch-Tung-Striebel Smoother """
    pass

class ExtendedKalmanSmoother(ExtendedRauchTungStriebelSmoother):
    """ Alias class for First Order Rauch-Tung-Striebel Smoother """
    pass

class FirstOrderExtendedRauchTungStriebelSmoother(ExtendedRauchTungStriebelSmoother):
    """ Alias class for First Order Rauch-Tung-Striebel Smoother """
    pass

class UnscentedKalmanSmoother(UnscentedRauchTungStriebelSmoother):
    """ Alias class for Unscented Rauch-Tung-Striebel Smoother """
    pass
