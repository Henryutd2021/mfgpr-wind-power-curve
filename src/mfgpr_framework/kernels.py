"""Covariance functions: the squared-exponential (RBF) kernel of Eq. (12) and its sums and products (Eq. 11).

Every kernel keeps its free parameters in ``theta`` (log variance, log length scales) and returns
the derivative of its covariance matrix with respect to each of them, so that the marginal
likelihood can be maximised with exact gradients (``gp.GaussianProcess``).

Eq. (12) writes the RBF kernel with automatic-relevance-determination weights w_i; here the kernel
is parameterised by length scales l_i, with w_i = 1 / l_i**2.
"""
from __future__ import annotations

import numpy as np

LOG_BOUNDS = (-12.0, 12.0)  # bounds on log-parameters during optimisation


class Kernel:
    """Base class. Subclasses implement ``K``, ``Kdiag``, ``gradients`` and the ``theta`` property."""

    def __add__(self, other: "Kernel") -> "Sum":
        return Sum(self, other)

    def __mul__(self, other: "Kernel") -> "Product":
        return Product(self, other)

    @property
    def n_params(self) -> int:
        return len(self.theta)

    def bounds(self) -> list[tuple[float, float]]:
        return [LOG_BOUNDS] * self.n_params


class RBF(Kernel):
    """Squared-exponential kernel on selected input columns (Eq. 12).

    c(x, x') = variance * exp(-1/2 * sum_i (x_i - x'_i)**2 / l_i**2), i over ``active_dims``.

    Parameters
    ----------
    active_dims : column indices of the input matrix the kernel acts on.
    variance, lengthscale : initial values; ``lengthscale`` is a scalar or one value per active column.
    fixed_variance : keep the variance at its initial value (used for the factor c^f of Eq. 11,
        where only the product of two variances is identifiable).
    lengthscale_bounds : optional (lower, upper) limits on the length scales during optimisation,
        scalars or one pair of values per active column; default: the generic log bounds.
    """

    def __init__(self, active_dims, variance: float = 1.0, lengthscale=1.0, fixed_variance: bool = False,
                 lengthscale_bounds=None):
        self.active_dims = tuple(int(d) for d in np.atleast_1d(active_dims))
        if not self.active_dims:
            raise ValueError("RBF needs at least one active input column")
        ls = np.broadcast_to(np.asarray(lengthscale, dtype=float), (len(self.active_dims),)).copy()
        if variance <= 0 or np.any(ls <= 0):
            raise ValueError("variance and length scales must be positive")
        self.fixed_variance = fixed_variance
        self._log_variance = float(np.log(variance))
        self._log_lengthscale = np.log(ls)
        if lengthscale_bounds is None:
            self._ls_bounds = [LOG_BOUNDS] * len(self.active_dims)
        else:
            lo, hi = (np.broadcast_to(np.asarray(b, dtype=float), (len(self.active_dims),)) for b in lengthscale_bounds)
            if np.any(lo <= 0) or np.any(hi <= lo):
                raise ValueError("length-scale bounds must satisfy 0 < lower < upper")
            self._ls_bounds = [(float(np.log(a)), float(np.log(b))) for a, b in zip(lo, hi)]
            self._log_lengthscale = np.clip(self._log_lengthscale, [b[0] for b in self._ls_bounds],
                                            [b[1] for b in self._ls_bounds])

    # parameters ---------------------------------------------------------------------------------
    @property
    def theta(self) -> np.ndarray:
        if self.fixed_variance:
            return self._log_lengthscale.copy()
        return np.concatenate([[self._log_variance], self._log_lengthscale])

    @theta.setter
    def theta(self, value) -> None:
        value = np.asarray(value, dtype=float)
        if self.fixed_variance:
            self._log_lengthscale = value.copy()
        else:
            self._log_variance = float(value[0])
            self._log_lengthscale = value[1:].copy()

    @property
    def param_names(self) -> list[str]:
        names = [f"log_lengthscale[{d}]" for d in self.active_dims]
        return names if self.fixed_variance else ["log_variance"] + names

    def bounds(self) -> list[tuple[float, float]]:
        return self._ls_bounds if self.fixed_variance else [LOG_BOUNDS] + self._ls_bounds

    @property
    def variance(self) -> float:
        return float(np.exp(self._log_variance))

    @property
    def lengthscale(self) -> np.ndarray:
        return np.exp(self._log_lengthscale)

    # covariance ---------------------------------------------------------------------------------
    def _scaled_sq_diff(self, X1, X2):
        a = np.asarray(X1, dtype=float)[:, self.active_dims] / self.lengthscale
        b = np.asarray(X2, dtype=float)[:, self.active_dims] / self.lengthscale
        return (a[:, None, :] - b[None, :, :]) ** 2  # n1 x n2 x d

    def K(self, X1, X2=None) -> np.ndarray:
        X2 = X1 if X2 is None else X2
        return self.variance * np.exp(-0.5 * self._scaled_sq_diff(X1, X2).sum(axis=2))

    def Kdiag(self, X) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.variance)

    def gradients(self, X) -> list[np.ndarray]:
        """dK/dtheta at the training inputs, in the order of ``theta``."""
        sq = self._scaled_sq_diff(X, X)
        K = self.variance * np.exp(-0.5 * sq.sum(axis=2))
        grads = [K * sq[:, :, i] for i in range(sq.shape[2])]  # d/d log l_i
        return grads if self.fixed_variance else [K] + grads


def _leaves(kernel: Kernel) -> list:
    return [leaf for part in kernel.parts for leaf in _leaves(part)] if isinstance(kernel, _Combination) else [kernel]


class _Combination(Kernel):
    def __init__(self, left: Kernel, right: Kernel):
        ids = [id(k) for k in _leaves(left) + _leaves(right)]
        if len(ids) != len(set(ids)):
            raise ValueError("a kernel instance can appear only once in a sum or product; use separate instances")
        self.parts = (left, right)

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([p.theta for p in self.parts])

    @theta.setter
    def theta(self, value) -> None:
        value = np.asarray(value, dtype=float)
        n = self.parts[0].n_params
        self.parts[0].theta, self.parts[1].theta = value[:n], value[n:]

    @property
    def param_names(self) -> list[str]:
        return [f"{i}.{name}" for i, p in enumerate(self.parts) for name in p.param_names]

    def bounds(self) -> list[tuple[float, float]]:
        return self.parts[0].bounds() + self.parts[1].bounds()


class Sum(_Combination):
    """c = c_left + c_right."""

    def K(self, X1, X2=None):
        return self.parts[0].K(X1, X2) + self.parts[1].K(X1, X2)

    def Kdiag(self, X):
        return self.parts[0].Kdiag(X) + self.parts[1].Kdiag(X)

    def gradients(self, X):
        return self.parts[0].gradients(X) + self.parts[1].gradients(X)


class Product(_Combination):
    """c = c_left * c_right (element-wise)."""

    def K(self, X1, X2=None):
        return self.parts[0].K(X1, X2) * self.parts[1].K(X1, X2)

    def Kdiag(self, X):
        return self.parts[0].Kdiag(X) * self.parts[1].Kdiag(X)

    def gradients(self, X):
        K0, K1 = self.parts[0].K(X), self.parts[1].K(X)
        return [g * K1 for g in self.parts[0].gradients(X)] + [K0 * g for g in self.parts[1].gradients(X)]
