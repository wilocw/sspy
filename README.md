# Bayesian State Space Modelling in Python

```sspy``` serves as a Python library to bring together implementations of solving state space models, particularly methods that learn system dynamics. The library is being implemented as part of an ongoing survey into learning state space dynamics. At present, the basic utilities for working with state space implementing filters and smoothers are being implemented. Later, implementations of papers will be added and benchmarked.

** This repository is primarily hosted and maintained on [GitLab](https://gitlab.com/wilocw/sspy) at with the [GitHub-hosted repository](https://github.com/wilocw/sspy) serving as a secondary remote. Issues, comments and pull requests may be made via either repository.**

The base of this library is inspired by the [EKF/UKF Toolbox for MATLAB](http://becs.aalto.fi/en/research/bayes/ekfukf/) by Särkkä, Hartikainen, and Solin.

## Currently implemented (v0.1.0)

### Filters
- Kalman filters
    - Linear-discrete KF
    - First-order extended KF
    - Second-order extended KF
    - Unscented KF

### Smoothers
- Rauch-Tung-Striebel (/Kalman) smoothers
    - Linear-discrete RTS
    - First-order extended RTS
    - Second-order extended RTS
    - Unscented RTS

### Utilities
- Dynamic function evaluation, taking method handles, lambda expressions or linear systems as numpy matrices and applying it to column vector and control
- Unscented transform
- Basic estimate plotting
- Noisy and noiseless system model generators

## Planned
This list is vague, while the review is in development.

- Sequential Monte Carlo / particle filters smoothers
- Kernel-embedded conditional distributions
- Gaussian process state space models
- "Deep" implementations
    - e.g. Kalman VAEs, Backprop KF
- Implicit models

## Related libraries
These libraries are more complete at present, but do not cover the intended scope.

- [PySSM: A Python Module for Bayesian Inference of Linear Gaussian State Space Models](https://bitbucket.org/christophermarkstrickland/pyssm)

- [FilterPy: Kalman filters and other optimal and non-optimal estimation filters in Python](https://github.com/rlabbe/filterpy)

- [EKF/UKF Toolbox for MATLAB](http://becs.aalto.fi/en/research/bayes/ekfukf/)
