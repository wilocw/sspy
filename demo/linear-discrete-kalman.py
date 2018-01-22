import numpy as np
from matplotlib import pyplot as plt

from sspy.filters import KalmanFilter
from sspy.smoothers import KalmanSmoother

from sspy.util import model_noisy, model_noiseless, plot_estimate, rmse

## Reseed random generator
_reseed = True

# Process model
F = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
# H
H = np.array([[1, 0, 0]])

# Process noise covariance
Q = np.diag(np.array([0., 0.01, 0.001]))

# Observation noise covariance
R = np.array([[400]])

# Initial state and estimation error covariance (assume Q)
x0 = np.array([[0., 0.1, 0.05]]).T
P0 = np.diag(np.array([10, 1, 0.1])) #Q.copy()

# Data dimensions
n_x = 3
n_y = 1

# Number of observations
n_t = 100
if _reseed:
    seed = np.random.randint(0,np.iinfo(np.int32).max)
    print(seed)
else:
    seed = 1892303842
np.random.seed(seed)

## Kalman filter subroutine
def run_filter(y_noisy):
    kf = KalmanFilter(x0, P0, F, Q, H, R, _verbose=False)

    x_estimate = np.zeros((n_x, n_t))
    P_estimate = np.zeros((n_x, n_x, n_t))
    y_estimate = np.zeros((n_y, n_t))
    Py_estimate = np.zeros((n_y, n_y, n_t))

    x_estimate[:,0] = x0.ravel()
    y_estimate[:,0] = (H @ x0).ravel()
    P_estimate[:,:,0] = P0
    Py_estimate[:,:,0] = H @ P0 @ H.T + R

    for i_ in range(1, n_t):
        kf.predict()
        kf.update(y_noisy[:,i_])
        x_estimate[:,i_] = kf.state['expected'].ravel()
        P_estimate[:,:,i_] = kf.state['err_cov']
        y_estimate[:,i_] = (H @ x_estimate[:,i_].reshape(n_x,1)).ravel()
        Py_estimate[:,:,i_] = H @ P_estimate[:,:,i_] @ H.T + R

    return x_estimate, P_estimate, y_estimate, Py_estimate, kf

## Kalman (Rauch-Tung-Streibel) smoother subroutine
def run_smoother(kf):
    rts = KalmanSmoother.from_filter(kf)
    states = rts.smooth()

    x_smoothed = np.zeros_like(x_true)
    P_smoothed = np.zeros((n_x, n_x, n_t))
    y_smoothed = np.zeros_like(y_true)
    Py_smoothed = np.zeros((n_y, n_y, n_t))

    x_smoothed[:,0] = x0.ravel()
    y_smoothed[:,0] = (H @ x0).ravel()
    P_smoothed[:,:,0] = P0

    for i_ in range(1, n_t):
        x_smoothed[:,i_] = states[i_]['expected'].ravel()
        P_smoothed[:,:,i_] = states[i_]['err_cov']
        y_smoothed[:,i_] = (H @ x_smoothed[:,i_].reshape(n_x,1)).ravel()
        Py_smoothed[:,:,i_] = H @ P_smoothed[:,:,i_] @ H.T + R

    return x_smoothed, P_smoothed, y_smoothed, Py_smoothed, rts

## Run Kalman prediction (n_s * n_t filter samples)
def run_predictor(y_noisy, ratio=0.8):
    pr = KalmanFilter(x0, P0, F, Q, H, R, _verbose=False)

    n_s = int(np.fix(ratio * n_t))

    x_predict = np.zeros((n_x, n_t-n_s))
    P_predict = np.zeros((n_x, n_x, n_t-n_s))
    y_predict = np.zeros((n_y, n_t-n_s))
    Py_predict = np.zeros((n_y, n_y, n_t-n_s))

    for i_ in range(1, n_s):#%n_t):
        pr.predict()
        pr.update(y_noisy[:,i_])

    for i_ in range(0,n_t-n_s):
        pr.predict()
        x_predict[:,i_]   = pr.state['expected'].ravel()
        P_predict[:,:,i_] = pr.state['err_cov']
        y_predict[:,i_] = (H @ x_predict[:,i_].reshape(n_x,1)).ravel()
        Py_predict[:,:,i_] = H @ P_predict[:,:,i_] @ H.T + R

    return x_predict, P_predict, y_predict, Py_predict

### PLOTTING
def plot_filter():
    f, ax = plt.subplots(3, 1, sharex='all')
    ax[0].set_title('Filter estimates')

    ax[0].plot(y_true.T, 'k-')
    ax[0].plot(y_noisy.T, 'b.')
    plot_estimate(np.arange(n_t), y_estimate, P_estimate[0,0,:], ax=ax[0])

    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    ax[0].legend(labels=['true state','noisy measurements','filtered estimate','estimate confidence (0.95)'])

    lbl = ['$\dot{x}$','$\ddot{x}$']
    for i in range(2):
        #ax[i+1].figure(figsize=(14,4))
        ax[i+1].plot(x_true[i+1,:].ravel(),'k-')

        plot_estimate(np.arange(n_t),
                      x_estimate[i+1,:], P_estimate[i+1,i+1,:],
                      ax=ax[i+1])

        ax[i+1].set_xlabel('$t$')
        ax[i+1].set_ylabel(lbl[i])
        ax[i+1].legend(labels=['true state','filtered estimate','estimate confidence (0.95)'])

def plot_smoother():
    f, ax = plt.subplots(3, 1, sharex='all')
    ax[0].set_title('Smoother estimate')
    ax[0].plot(y_true.T, 'k-')
    ax[0].plot(y_estimate.T, 'r--', lw=3)

    plot_estimate(np.arange(n_t), y_smoothed, P_smoothed[0,0,:], c='b',ax=ax[0])

    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    ax[0].legend(labels=['true state','filtered estimate','smoothed estimate','smoother confidence (0.95)'])

    lbl = ['$\dot{x}$','$\ddot{x}$']
    for i in range(2):
        #plt.figure(figsize=(14,4))
        ax[i+1].plot(x_true[i+1,:].ravel(),'k-')
        ax[i+1].plot(x_estimate[i+1,:].ravel(), 'r--', lw=3)

        plot_estimate(np.arange(n_t), x_smoothed[i+1,:], P_smoothed[i+1,i+1,:], c='b',ax=ax[i+1])

        ax[i+1].set_xlabel('$t$')
        ax[i+1].set_ylabel(lbl[i])
        ax[i+1].legend(labels=['true state','filtered estimate','smoothed estimate','smoother confidence (0.95)'])

def plot_prediction(r):
    plt.figure()
    plt.title('Predictor (%3.2f)' % r)

    plt.plot(y_true.T, 'k-')
    plt.plot(y_noisy[:,0:n_s+1].T, 'b.')

    plot_estimate(np.arange(n_s+1), y_estimate[:,0:n_s+1], P_estimate[0,0,0:n_s+1])

    plot_estimate(np.arange(n_s,n_t), y_predict, P_predict[0,0,:], c='g')

    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.legend(labels=['true state', 'noisy measurements', 'filtered estimate','prediction','estimation confidence (0.95)','prediction confidence (0.95)'])


### RUN TESTS
num_test = 1000
filter_ratios = [0.9, 0.75, 0.5, 0.2, 0.]

f_rmse = np.zeros((n_x + n_y, num_test))
s_rmse = np.zeros((n_x + n_y, num_test))
p_rmse = np.zeros((n_x + n_y, num_test, len(filter_ratios)))

for k in range(num_test):
    if not k % 10:
        print(k)
    # Model systems with noise
    x_true, y_true, y_noisy = model_noisy(x0, F, Q, None, H, R, n=n_t)

    # Run filter
    x_estimate, P_estimate, y_estimate, Py_estimate, kf = run_filter(y_noisy)
    # Run smoother
    x_smoothed, P_smoothed, y_smoothed, Py_smoothed, _ = run_smoother(kf)
    # Calculate RMSE
    f_rmse[:,k] = rmse(np.vstack([x_true,y_true]),
                       np.vstack([x_estimate, y_estimate])).ravel()
    s_rmse[:,k] = rmse(np.vstack([x_true,y_true]),
                       np.vstack([x_smoothed, y_smoothed])).ravel()

    if k is num_test-1:
        plot_filter()
        plot_smoother()

    for j, r_ in enumerate(filter_ratios):
        x_predict, P_predict, y_predict, Py_predict = run_predictor(y_noisy, r_)

        n_s = int(np.fix(r_ * n_t))
        p_rmse[:,k,j] = rmse(np.vstack([x_true[:,n_s:n_t],y_true[:,n_s:n_t]]),
                             np.vstack([x_predict, y_predict])).ravel()
        if k is num_test-1 and (j is 1 or j is 2):
            plot_prediction(r_)

## Show results
f_m, f_s = np.mean(f_rmse, 1), np.std(f_rmse, 1)
s_m, s_s = np.mean(s_rmse, 1), np.std(s_rmse, 1)
p_m, p_s = np.mean(p_rmse, 1).squeeze(), np.std(p_rmse, 1).squeeze()


lbl = ['x[0]','x[1]','x[2]','y[0]']
print('        RMSE             ||   prediction ')
print('_____|_filter__|_smoother||___0.90___|___0.75___|___0.50___|___0.20___|___0.00___')
for i in range(4):
    print('%s | %6.5f | %6.5f || %8.4f | %8.4f | %8.4f | %8.4f | %8.4f' %\
            (lbl[i],f_m[i],s_m[i],p_m[i,0],p_m[i,1],p_m[i,2],p_m[i,3],p_m[i,4]))
    print(' +/- | %6.5f | %6.5f || %8.4f | %8.4f | %8.4f | %8.4f | %8.4f'  %\
                (f_s[i], s_s[i],p_s[i,0],p_s[i,1],p_s[i,2],p_s[i,3],p_s[i,4]))
    print('=================================================================================')
# Plot all
plt.show()
