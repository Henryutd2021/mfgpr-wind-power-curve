"""Multi-fidelity Gaussian process models (Section 2.3).

* ``LinearMultiFidelityGP``: the autoregressive scheme of Eq. (6),
      g_t(x) = rho_{t-1} * g_{t-1}(x) + delta_t(x),
  with independent GPs delta_t and constant scaling factors rho. All levels are modelled jointly
  and every hyperparameter (kernels, rho, one noise variance per level) is fitted by maximising the
  joint marginal likelihood.
* ``NonlinearMultiFidelityGP``: the recursive scheme of Eq. (10),
      f_t(x) = g_t(x, f*_{t-1}(x)),
  with the structured kernel of Eq. (11),
      c_t = c_rho(x, x') * c_f(f*_{t-1}(x), f*_{t-1}(x')) + c_delta(x, x'),
  and RBF factors (Eq. 12). Level t is trained on the inputs [x, mean of level t-1 at x]; at
  prediction time the uncertainty of level t-1 is propagated by Monte Carlo sampling.

Levels are ordered from the lowest fidelity (index 0) to the highest (index T-1). Inputs are
n x d arrays (d = 1 for a power curve: wind speed).
"""
from __future__ import annotations

import numpy as np

from .gp import GaussianNoise, GaussianProcess, LevelNoise
from .kernels import RBF, Kernel

RHO_BOUNDS = (-10.0, 10.0)


def as_2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _check_inputs(X, n_inputs: int) -> np.ndarray:
    X = as_2d(X)
    if X.ndim != 2 or X.shape[1] != n_inputs:
        raise ValueError(f"inputs must have {n_inputs} column(s)")
    if not np.all(np.isfinite(X)):
        raise ValueError("inputs must be finite")
    return X


def _check_level(level, n_levels: int) -> int:
    if level is None:
        return n_levels - 1
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or not 0 <= level < n_levels:
        raise ValueError(f"level must be an integer between 0 and {n_levels - 1}")
    return int(level)


def _check_levels(levels):
    if len(levels) < 2:
        raise ValueError("a multi-fidelity model needs at least two levels")
    out = []
    for t, (X, y) in enumerate(levels):
        X, y = as_2d(X), np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError(f"level {t}: X and y must have the same, non-zero number of rows")
        out.append((X, y))
    if len({X.shape[1] for X, _ in out}) != 1:
        raise ValueError("all levels must have the same number of input columns")
    return out


def default_rbf(X, y, active_dims=None, fixed_variance: bool = False) -> RBF:
    """RBF kernel with data-based starting values and length-scale limits.

    Start: variance = var(y), length scale = 20 % of the input range. Limits: the length scale of each
    column stays between the median spacing of its distinct training values and 100 times its range.
    The lower limit keeps the optimiser away from the degenerate optimum in which a vanishing length
    scale turns the kernel into white noise.
    """
    X = as_2d(X)
    dims = list(range(X.shape[1])) if active_dims is None else list(active_dims)
    span = np.ptp(X[:, dims], axis=0)
    lo, hi = [], []
    for k, d in enumerate(dims):
        values = np.unique(X[:, d])
        if len(values) < 2 or span[k] <= 0:
            lo.append(np.exp(-12.0)), hi.append(np.exp(12.0))
            continue
        lo.append(float(np.median(np.diff(values))))
        hi.append(100.0 * float(span[k]))
    ls = np.maximum(0.2 * span, 1e-3)
    var = float(np.var(y)) if np.var(y) > 0 else 1.0
    return RBF(dims, variance=1.0 if fixed_variance else var, lengthscale=ls, fixed_variance=fixed_variance,
               lengthscale_bounds=(lo, hi))


# ------------------------------------------------------------------------------------------------
# Linear (autoregressive) model, Eq. (6)
# ------------------------------------------------------------------------------------------------
class LinearMultiFidelityKernel(Kernel):
    """Joint covariance of all levels under Eq. (6).

    With f_t = rho_{t-1} f_{t-1} + delta_t and f_0 = delta_0,
        cov(f_t(x), f_s(x')) = sum_{j <= min(t, s)} a_{t,j} a_{s,j} c_j(x, x'),
        a_{t,j} = prod_{i=j}^{t-1} rho_i   (empty product = 1).
    The level index is the last input column; ``base`` kernels act on the other columns.
    """

    def __init__(self, base: list[Kernel], rho=None):
        self.base = list(base)
        T = len(self.base)
        self.rho = np.ones(T - 1) if rho is None else np.asarray(rho, dtype=float).copy()
        if self.rho.shape != (T - 1,):
            raise ValueError("rho needs one value per pair of consecutive levels")

    @property
    def n_levels(self) -> int:
        return len(self.base)

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([k.theta for k in self.base] + [self.rho])

    @theta.setter
    def theta(self, value) -> None:
        value = np.asarray(value, dtype=float)
        pos = 0
        for k in self.base:
            k.theta = value[pos:pos + k.n_params]
            pos += k.n_params
        self.rho = value[pos:].copy()

    @property
    def param_names(self) -> list[str]:
        names = [f"delta[{t}].{n}" for t, k in enumerate(self.base) for n in k.param_names]
        return names + [f"rho[{i}]" for i in range(self.n_levels - 1)]

    def bounds(self):
        return [b for k in self.base for b in k.bounds()] + [RHO_BOUNDS] * (self.n_levels - 1)

    def _coefficients(self):
        """a[t, j] and da[i, t, j] = d a[t, j] / d rho_i."""
        T = self.n_levels
        a = np.zeros((T, T))
        da = np.zeros((max(T - 1, 0), T, T))
        for t in range(T):
            for j in range(t + 1):
                idx = list(range(j, t))
                a[t, j] = np.prod(self.rho[idx]) if idx else 1.0
                for i in idx:
                    da[i, t, j] = np.prod([self.rho[k] for k in idx if k != i]) if len(idx) > 1 else 1.0
        return a, da

    @staticmethod
    def _levels(X):
        return np.asarray(X)[:, -1].astype(int)

    def K(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        a, _ = self._coefficients()
        l1, l2 = self._levels(X1), self._levels(X2)
        return sum(np.outer(a[l1, j], a[l2, j]) * k.K(X1, X2) for j, k in enumerate(self.base))

    def Kdiag(self, X):
        a, _ = self._coefficients()
        lev = self._levels(X)
        return sum(a[lev, j] ** 2 * k.Kdiag(X) for j, k in enumerate(self.base))

    def gradients(self, X):
        a, da = self._coefficients()
        lev = self._levels(X)
        grads, Ks = [], []
        for j, k in enumerate(self.base):
            A = np.outer(a[lev, j], a[lev, j])
            grads += [A * g for g in k.gradients(X)]
            Ks.append(k.K(X))
        for i in range(self.n_levels - 1):
            g = np.zeros((len(lev), len(lev)))
            for j, Kj in enumerate(Ks):
                dA = np.outer(da[i, lev, j], a[lev, j]) + np.outer(a[lev, j], da[i, lev, j])
                g += dA * Kj
            grads.append(g)
        return grads


class LinearMultiFidelityGP:
    """Autoregressive multi-fidelity GP (Eq. 6), fitted jointly over all levels."""

    def __init__(self, noise_variance: float = 1e-2, jitter: float = 1e-8):
        self.noise_variance = noise_variance
        self.jitter = jitter
        self.gp: GaussianProcess | None = None
        self.n_levels = 0
        self.n_inputs = 0

    @staticmethod
    def stack(levels):
        """Stack level data into one input matrix with the level index as last column."""
        X = np.vstack([np.hstack([Xt, np.full((len(Xt), 1), t)]) for t, (Xt, _) in enumerate(levels)])
        y = np.concatenate([yt for _, yt in levels])
        return X, y

    def fit(self, levels, restarts: int = 5, seed=None):
        levels = _check_levels(levels)
        self.n_levels, self.n_inputs = len(levels), levels[0][0].shape[1]
        dims = list(range(self.n_inputs))
        base = [default_rbf(Xt, yt, dims) for Xt, yt in levels]
        kernel = LinearMultiFidelityKernel(base)
        noise = LevelNoise(self.n_levels, self.noise_variance * np.array([max(np.var(yt), 1e-12) for _, yt in levels]))
        self.gp = GaussianProcess(kernel, noise, jitter=self.jitter)
        X, y = self.stack(levels)
        self.gp.fit(X, y, restarts=restarts, seed=seed)
        return self

    @property
    def rho(self) -> np.ndarray:
        return self.gp.kernel.rho.copy()

    def predict(self, X, level: int | None = None, include_noise: bool = False):
        """Mean and variance of level ``level`` (default: the highest) at inputs ``X``."""
        if self.gp is None:
            raise RuntimeError("fit the model before predicting")
        level = _check_level(level, self.n_levels)
        X = _check_inputs(X, self.n_inputs)
        Xl = np.hstack([X, np.full((len(X), 1), level)])
        return self.gp.predict(Xl, include_noise=include_noise)


# ------------------------------------------------------------------------------------------------
# Nonlinear (recursive) model, Eqs. (10)-(12)
# ------------------------------------------------------------------------------------------------
class NonlinearMultiFidelityGP:
    """Recursive nonlinear multi-fidelity GP (Eqs. 10-12) with Monte-Carlo uncertainty propagation.

    Parameters
    ----------
    n_samples : number of Monte-Carlo samples of level t-1 used to predict level t.
    noise_variance : starting value of each level's noise variance, as a fraction of var(y_t).
    """

    def __init__(self, n_samples: int = 100, noise_variance: float = 1e-2, jitter: float = 1e-8):
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        self.n_samples = n_samples
        self.noise_variance = noise_variance
        self.jitter = jitter
        self.gps: list[GaussianProcess] = []
        self._eps: list[np.ndarray] = []
        self.n_inputs = 0

    @property
    def n_levels(self) -> int:
        return len(self.gps)

    def _level_kernel(self, Z, y):
        d = self.n_inputs
        x_dims = list(range(d))
        c_rho = default_rbf(Z, y, x_dims)
        c_f = default_rbf(Z, y, [d], fixed_variance=True)
        c_delta = default_rbf(Z, 0.1 * y, x_dims)
        return c_rho * c_f + c_delta

    def fit(self, levels, restarts: int = 5, seed=None):
        levels = _check_levels(levels)
        self.n_inputs = levels[0][0].shape[1]
        rng = np.random.default_rng(seed)
        self.gps, self._eps = [], []
        for t, (Xt, yt) in enumerate(levels):
            noise = GaussianNoise(self.noise_variance * max(np.var(yt), 1e-12))
            if t == 0:
                gp = GaussianProcess(default_rbf(Xt, yt), noise, jitter=self.jitter)
                gp.fit(Xt, yt, restarts=restarts, seed=rng.integers(2**32))
            else:
                prev_mean, _ = self.predict(Xt, level=t - 1)
                Z = np.hstack([Xt, prev_mean.reshape(-1, 1)])
                gp = GaussianProcess(self._level_kernel(Z, yt), noise, jitter=self.jitter)
                gp.fit(Z, yt, restarts=restarts, seed=rng.integers(2**32))
            self.gps.append(gp)
            # fixed standard-normal numbers for sampling this level when predicting the next one
            self._eps.append(rng.standard_normal(self.n_samples))
        return self

    def predict_components(self, X, level: int | None = None):
        """Monte-Carlo components of the latent prediction of level ``level`` (default: the highest).

        Level 0 is an ordinary GP (one component). For level t >= 1, S samples of f_{t-1}(x) are
        drawn with fixed standard-normal numbers (so predictions are deterministic once fitted), and
        level t is evaluated at [x, sample] for each. Returns (means, variances), both S x n: the
        predictive distribution is the equal-weight mixture of the S Gaussians.
        """
        if not self.gps:
            raise RuntimeError("fit the model before predicting")
        level = _check_level(level, self.n_levels)
        X = _check_inputs(X, self.n_inputs)
        mean, var = self.gps[0].predict(X)
        means, variances = mean[None, :], var[None, :]
        samples = mean[None, :] + np.sqrt(var)[None, :] * self._eps[0][:, None]  # S x n
        for t in range(1, level + 1):
            means = np.empty_like(samples)
            variances = np.empty_like(samples)
            for s in range(samples.shape[0]):
                means[s], variances[s] = self.gps[t].predict(np.hstack([X, samples[s][:, None]]))
            samples = means + np.sqrt(variances) * self._eps[t][:, None]
        return means, variances

    def predict(self, X, level: int | None = None, include_noise: bool = False):
        """Mean and variance of level ``level`` (default: the highest) at inputs ``X``.

        The first two moments of the Monte-Carlo mixture (``predict_components``), by the law of
        total variance: mean = average of the S means, var = average of the S variances + variance
        of the S means. ``include_noise`` adds the noise variance of that level.
        """
        level = _check_level(level, self.n_levels) if self.gps else level
        means, variances = self.predict_components(X, level)
        mean = means.mean(axis=0)
        var = variances.mean(axis=0) + means.var(axis=0)
        if include_noise:
            var = var + self.gps[level].noise.diag(mean[:, None])
        return mean, var
