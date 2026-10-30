"""Gaussian process regression (Section 2.2, Eqs. 4-5) with maximum-marginal-likelihood hyperparameters.

The GP has zero prior mean. Observations are y_i = f(x_i) + e_i with Gaussian noise e_i. The noise
model is either one variance (``GaussianNoise``) or one variance per fidelity level
(``LevelNoise``, used by the joint model of ``multifidelity.LinearMultiFidelityGP``).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .kernels import LOG_BOUNDS, Kernel

NOISE_BOUNDS = (np.log(1e-8), np.log(1e4))


class GaussianNoise:
    """One noise variance sigma_n**2 for all observations."""

    def __init__(self, variance: float = 1e-2, fixed: bool = False):
        if variance <= 0:
            raise ValueError("noise variance must be positive")
        self._log_variance = float(np.log(variance))
        self.fixed = fixed

    @property
    def theta(self) -> np.ndarray:
        return np.empty(0) if self.fixed else np.array([self._log_variance])

    @theta.setter
    def theta(self, value) -> None:
        if not self.fixed:
            self._log_variance = float(np.asarray(value)[0])

    @property
    def param_names(self) -> list[str]:
        return [] if self.fixed else ["noise.log_variance"]

    def bounds(self):
        return [] if self.fixed else [NOISE_BOUNDS]

    @property
    def variance(self) -> float:
        return float(np.exp(self._log_variance))

    def diag(self, X) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.variance)

    def gradients(self, X) -> list[np.ndarray]:
        return [] if self.fixed else [self.diag(X)]


class LevelNoise:
    """One noise variance per fidelity level; the level index is stored in input column ``level_col``."""

    def __init__(self, n_levels: int, variance=1e-2, level_col: int = -1):
        v = np.broadcast_to(np.asarray(variance, dtype=float), (n_levels,)).copy()
        if np.any(v <= 0):
            raise ValueError("noise variances must be positive")
        self.n_levels = n_levels
        self.level_col = level_col
        self._log_variance = np.log(v)

    @property
    def theta(self) -> np.ndarray:
        return self._log_variance.copy()

    @theta.setter
    def theta(self, value) -> None:
        self._log_variance = np.asarray(value, dtype=float).copy()

    @property
    def param_names(self) -> list[str]:
        return [f"noise[{t}].log_variance" for t in range(self.n_levels)]

    def bounds(self):
        return [NOISE_BOUNDS] * self.n_levels

    @property
    def variance(self) -> np.ndarray:
        return np.exp(self._log_variance)

    def levels(self, X) -> np.ndarray:
        return np.asarray(X)[:, self.level_col].astype(int)

    def diag(self, X) -> np.ndarray:
        return self.variance[self.levels(X)]

    def gradients(self, X) -> list[np.ndarray]:
        lev = self.levels(X)
        return [np.where(lev == t, self.variance[t], 0.0) for t in range(self.n_levels)]


class GaussianProcess:
    """Zero-mean GP regression.

    Posterior mean (Eq. 4) and variance (Eq. 5) at new inputs x:
        mean(x) = c_*^T (C + Sigma_n)^-1 y
        var(x)  = c(x, x) - c_*^T (C + Sigma_n)^-1 c_*
    where Sigma_n is the (diagonal) noise covariance. ``predict(..., include_noise=True)`` adds the
    noise variance, giving the predictive distribution of a new observation y rather than of f(x).
    """

    def __init__(self, kernel: Kernel, noise=None, jitter: float = 1e-8):
        self.kernel = kernel
        self.noise = noise if noise is not None else GaussianNoise()
        self.jitter = jitter
        self.X = None
        self.y = None
        self._chol = None
        self._alpha = None
        self._factor_theta = None  # parameters at the last factorisation

    # parameters ---------------------------------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.kernel.theta, self.noise.theta])

    @theta.setter
    def theta(self, value) -> None:
        value = np.asarray(value, dtype=float)
        n = self.kernel.n_params
        self.kernel.theta = value[:n]
        self.noise.theta = value[n:]

    @property
    def param_names(self) -> list[str]:
        return list(self.kernel.param_names) + list(self.noise.param_names)

    def bounds(self):
        return self.kernel.bounds() + self.noise.bounds()

    # likelihood ---------------------------------------------------------------------------------
    def _covariance(self, X):
        return self.kernel.K(X) + np.diag(self.noise.diag(X) + self.jitter)

    def neg_log_marginal_likelihood(self, theta=None, X=None, y=None, gradient: bool = True):
        """-log p(y | X, theta) and its gradient with respect to ``theta``.

        Evaluating at a trial ``theta`` leaves the model's own parameters unchanged.
        """
        if theta is not None:
            saved = self.theta
            theta = np.asarray(theta, dtype=float).ravel()
            if theta.shape != saved.shape:
                raise ValueError(f"theta must have {len(saved)} values")
            try:
                self.theta = theta
                return self.neg_log_marginal_likelihood(None, X, y, gradient)
            finally:
                self.theta = saved
        X = self.X if X is None else X
        y = self.y if y is None else y
        K = self._covariance(X)
        try:
            chol = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return (np.inf, np.zeros(len(self.theta))) if gradient else np.inf
        alpha = cho_solve(chol, y, check_finite=False)
        nll = 0.5 * y @ alpha + np.log(np.diag(chol[0])).sum() + 0.5 * len(y) * np.log(2 * np.pi)
        if not gradient:
            return nll
        Kinv = cho_solve(chol, np.eye(len(y)), check_finite=False)
        W = np.outer(alpha, alpha) - Kinv
        grads = self.kernel.gradients(X)
        grad_k = [-0.5 * np.sum(W * g) for g in grads]
        grad_n = [-0.5 * np.sum(np.diag(W) * d) for d in self.noise.gradients(X)]
        return nll, np.array(grad_k + grad_n)

    # fitting ------------------------------------------------------------------------------------
    def fit(self, X, y, restarts: int = 5, seed=None, optimize: bool = True):
        """Store the training data and maximise the marginal likelihood from ``restarts`` + 1 starts.

        The first start is the current ``theta`` (moved inside the bounds); the others add
        standard-normal noise to it (log-parameters on the log scale, scaling factors on their own
        scale), seeded by ``seed``, and are moved inside the bounds too. The best optimum is kept.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be n x d and y must have n values")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("training data must be finite")
        self.X, self.y = X, y
        if optimize and len(self.theta):
            rng = np.random.default_rng(seed)
            bounds = self.bounds()
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            start = np.clip(self.theta, lo, hi)
            best = (self.neg_log_marginal_likelihood(start, gradient=False), start)
            for r in range(restarts + 1):
                x0 = start if r == 0 else np.clip(start + rng.standard_normal(len(start)), lo, hi)
                res = minimize(self.neg_log_marginal_likelihood, x0, jac=True, method="L-BFGS-B",
                               bounds=bounds)
                if np.isfinite(res.fun) and res.fun < best[0]:
                    best = (float(res.fun), res.x)
            self.theta = best[1]
        self._factorize()
        return self

    def _factorize(self):
        K = self._covariance(self.X)
        self._chol = cho_factor(K, lower=True, check_finite=False)
        self._alpha = cho_solve(self._chol, self.y, check_finite=False)
        self._factor_theta = self.theta

    @property
    def log_marginal_likelihood(self) -> float:
        return -float(self.neg_log_marginal_likelihood(gradient=False))

    # prediction ---------------------------------------------------------------------------------
    def predict(self, Xnew, include_noise: bool = False, chunk: int = 2048):
        """Posterior mean and variance (Eqs. 4-5) at ``Xnew``; optionally plus the noise variance.

        If the parameters were changed after fitting, the covariance is factorised again first.
        Rows are processed in chunks of ``chunk`` to bound memory.
        """
        if self.X is None:
            raise RuntimeError("fit the model before predicting")
        if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
            raise ValueError("chunk must be a positive integer")
        if self._chol is None or not np.array_equal(self.theta, self._factor_theta):
            self._factorize()
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim != 2 or Xnew.shape[1] != self.X.shape[1]:
            raise ValueError(f"inputs must be an n x {self.X.shape[1]} array")
        if not np.all(np.isfinite(Xnew)):
            raise ValueError("inputs must be finite")
        mean = np.empty(len(Xnew))
        var = np.empty(len(Xnew))
        for a in range(0, len(Xnew), chunk):
            Xc = Xnew[a:a + chunk]
            Ks = self.kernel.K(Xc, self.X)
            mean[a:a + chunk] = Ks @ self._alpha
            v = cho_solve(self._chol, Ks.T, check_finite=False)
            var[a:a + chunk] = self.kernel.Kdiag(Xc) - np.einsum("ij,ji->i", Ks, v)
        var = np.maximum(var, 0.0)
        if include_noise:
            var = var + self.noise.diag(Xnew)
        return mean, var


__all__ = ["GaussianNoise", "LevelNoise", "GaussianProcess", "LOG_BOUNDS"]
