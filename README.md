# Efficient Active Manifold Identification via Accelerated Iteratively Reweighted Nuclear Norm Minimization

This Matlab repo is for an iteratively reweighted nuclear norm minimization.


Numerical implementation for solving the nonconvex Schatten-p norm regularization problem:

## The problem
We focus on the matrics optimization problem with Schatten-p norm regularizer

$$
\min_{X} \quad f(X)+ \lambda \|X\|_{p}^{p} 
$$


## Usage
using the function in 'fun/EIRNRI.m'

## Input Description
for `EIRNRI`:
INPUT ::
- X0: start point
- M: the observation matrix
- sp: the Schatten-p norm
- lambda: regularization parameter
- mask: mask matrix
- tol: reconstruction error tolerance
- options
  - max_iter, maxibetam number of iterations, default = 2000
  - eps, epsilon, default = 1
  - beta, proximal parameter, default = 1.1
  - KLopt, termination conditions, default = $10^{-5}\min$ ( size $(M)$ )
  - mu, the scale factor for `eps` , default = 0.7
  - alpha, extrapolation factor, default = 0.1
  - Rel, the options for record the correlation distance
  - zero, the options for trunk 0, default = $10^{-16}$
  - teps, the options for trunk eps, default = $10^{-16}$
  - objf, string of the objective function, default: $f(x) = \frac{1}{2}\|x-P_{\Omega}(x)\|_{F}^{2}$
  - Gradf, string of the first order derivative, default

## Input Description
OUTPUT ::
- Par
  - Xsol: restored matrix
