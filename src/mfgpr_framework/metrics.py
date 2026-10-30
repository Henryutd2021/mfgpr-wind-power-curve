"""Evaluation metrics (Sections 3.2-3.3, Eqs. 13-20) and interval diagnostics.

Deterministic metrics compare a point estimate x_hat with actual values x (Eqs. 13-17).
Probabilistic metrics take a Gaussian predictive distribution N(mean, std**2) per record:
pinball loss (Eq. 18), sharpness (Eq. 19), reliability (Eq. 20), and two curves over nominal
proportions (central-interval width and observed coverage). For the nonlinear multi-fidelity model
this Gaussian matches the first two moments of its Monte-Carlo mixture; ``mixture_cdf`` with
``reliability_numeric`` evaluates Eq. (20) on the mixture itself.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm


def _pair(actual, predicted):
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must have the same, non-zero length")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
        raise ValueError("actual and predicted values must be finite")
    return a, p


# deterministic, Eqs. (13)-(17) -------------------------------------------------------------------
def rmse(actual, predicted) -> float:
    """Eq. (13)."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mae(actual, predicted) -> float:
    """Eq. (15)."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(p - a)))


def nrmse(actual, predicted, x_max: float) -> float:
    """Eq. (14): RMSE / x_max (dimensionless)."""
    return rmse(actual, predicted) / _positive(x_max)


def nmae(actual, predicted, x_max: float) -> float:
    """Eq. (16): MAE / x_max (dimensionless)."""
    return mae(actual, predicted) / _positive(x_max)


def mape(actual, predicted, min_actual: float = 0.0) -> float:
    """Eq. (17), in percent, over the records with |x| > ``min_actual`` (x = 0 has no relative error)."""
    a, p = _pair(actual, predicted)
    keep = np.abs(a) > min_actual
    if not keep.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs((p[keep] - a[keep]) / a[keep])))


def _positive(x):
    if not (np.isfinite(x) and x > 0):
        raise ValueError("x_max must be positive and finite")
    return float(x)


def deterministic_metrics(actual, predicted, x_max: float, min_actual: float = 0.0) -> dict:
    """All of Eqs. (13)-(17) in one dictionary."""
    return {
        "rmse": rmse(actual, predicted),
        "nrmse": nrmse(actual, predicted, x_max),
        "mae": mae(actual, predicted),
        "nmae": nmae(actual, predicted, x_max),
        "mape_percent": mape(actual, predicted, min_actual),
    }


# probabilistic -----------------------------------------------------------------------------------
def _gaussian(y, mean, std):
    y = np.asarray(y, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()
    std = _std(std).ravel()
    if not (y.shape == mean.shape == std.shape) or y.size == 0:
        raise ValueError("y, mean and std must have the same, non-zero length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(mean))):
        raise ValueError("y and mean must be finite")
    return y, mean, std


def _std(std):
    std = np.asarray(std, dtype=float)
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("std must be positive and finite")
    return std


def _probability(p, name):
    if not (np.isfinite(p) and 0 < p < 1):
        raise ValueError(f"{name} must be in (0, 1)")
    return float(p)


def gaussian_quantile(mean, std, tau):
    """tau-quantile of N(mean, std**2)."""
    tau = _probability(tau, "tau")
    try:
        mean, std = np.broadcast_arrays(np.asarray(mean, dtype=float), _std(std))
    except ValueError:
        raise ValueError("mean and std must have the same or broadcastable shapes") from None
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite")
    return mean + std * norm.ppf(tau)


def pinball_loss(y, q_pred, tau: float) -> float:
    """Eq. (18), averaged over records: (y - q) tau if y >= q, else (q - y)(1 - tau)."""
    tau = _probability(tau, "tau")
    y, q = _pair(y, q_pred)
    d = y - q
    return float(np.mean(np.where(d >= 0, d * tau, -d * (1 - tau))))


def mean_pinball_loss(y, mean, std, quantiles=np.arange(0.05, 0.96, 0.05)) -> float:
    """Eq. (18) averaged over records and over the quantile levels ``quantiles``."""
    y, mean, std = _gaussian(y, mean, std)
    return float(np.mean([pinball_loss(y, gaussian_quantile(mean, std, t), t) for t in quantiles]))


def bin_probabilities(mean, std, edges) -> np.ndarray:
    """N x M matrix of forecast probabilities of M categories (bins between ``edges``).

    The two outer bins absorb the tails, so each row sums to 1.
    """
    mean = np.asarray(mean, dtype=float).ravel()
    std = _std(std).ravel()
    if mean.shape != std.shape or not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite and match std")
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 3 or np.any(np.isnan(edges)) or not np.all(np.isfinite(edges[1:-1]))
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("edges must be increasing with at least three values; only the outer two may be infinite")
    cdf = norm.cdf((edges[None, 1:-1] - mean[:, None]) / std[:, None])
    cdf = np.hstack([np.zeros((len(mean), 1)), cdf, np.ones((len(mean), 1))])
    return np.diff(cdf, axis=1)


def sharpness(probabilities) -> float:
    """Eq. (19): S = 1/N * sum_i sum_j (p_ij - pbar_j)**2 for an N x M probability matrix."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.size == 0:
        raise ValueError("probabilities must be an N x M matrix")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.sum((p - p.mean(axis=0)) ** 2) / p.shape[0])


def reliability_gaussian(y, mean, std) -> float:
    """Eq. (20) with dP(z) = dz for Gaussian forecasts (closed form):

    integral of (F(z) - 1{y <= z})**2 dz = std * [u (2 Phi(u) - 1) + 2 phi(u) - 1/sqrt(pi)],
    u = (y - mean) / std; averaged over records.
    """
    y, mean, std = _gaussian(y, mean, std)
    u = (y - mean) / std
    return float(np.mean(std * (u * (2 * norm.cdf(u) - 1) + 2 * norm.pdf(u) - 1 / np.sqrt(np.pi))))


def reliability_numeric(y, cdf, grid) -> float:
    """Eq. (20) for any forecast CDF, by the trapezoidal rule on ``grid`` (dP(z) = dz).

    ``cdf(z)`` must return an N x len(grid) matrix of forecast CDF values for the N records.
    """
    y = np.asarray(y, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float)
    F = np.asarray(cdf(grid), dtype=float)
    if F.shape != (len(y), len(grid)):
        raise ValueError("cdf(grid) must return an N x len(grid) matrix")
    H = (y[:, None] <= grid[None, :]).astype(float)
    return float(np.mean(trapezoid((F - H) ** 2, grid, axis=1)))


def mixture_cdf(z, means, stds) -> np.ndarray:
    """CDF of equal-weight Gaussian mixtures at the points ``z``: an n x len(z) matrix.

    ``means`` and ``stds`` are S x n (S components for each of n records), as returned by
    ``NonlinearMFGPR.predict_components``. Use with ``reliability_numeric`` for Eq. (20).
    """
    means = np.atleast_2d(np.asarray(means, dtype=float))
    stds = np.atleast_2d(np.asarray(stds, dtype=float))
    if means.shape != stds.shape or not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        raise ValueError("means and stds must be finite S x n arrays of the same shape, stds > 0")
    z = np.asarray(z, dtype=float).ravel()
    return norm.cdf((z[None, None, :] - means[:, :, None]) / stds[:, :, None]).mean(axis=0)


def interval_width(std, nominal) -> float:
    """Mean width of the central prediction interval with probability ``nominal``."""
    nominal = _probability(nominal, "nominal")
    return float(np.mean(2 * norm.ppf(0.5 + nominal / 2) * _std(std)))


def interval_coverage(y, mean, std, nominal) -> float:
    """Share of records inside the central prediction interval with probability ``nominal``."""
    nominal = _probability(nominal, "nominal")
    y, mean, std = _gaussian(y, mean, std)
    half = norm.ppf(0.5 + nominal / 2) * std
    return float(np.mean(np.abs(y - mean) <= half))


def interval_curves(y, mean, std, nominal=np.arange(0.1, 0.91, 0.1)):
    """Width and coverage of central intervals for each nominal proportion (rows)."""
    return [{"nominal": float(n), "mean_width": interval_width(std, n),
             "coverage": interval_coverage(y, mean, std, n)} for n in nominal]


def probabilistic_metrics(y, mean, std, edges, quantiles=np.arange(0.05, 0.96, 0.05)) -> dict:
    """Pinball loss (Eq. 18), sharpness (Eq. 19) and reliability (Eq. 20) in one dictionary."""
    y, mean, std = _gaussian(y, mean, std)
    return {
        "pinball": mean_pinball_loss(y, mean, std, quantiles),
        "sharpness_eq19": sharpness(bin_probabilities(mean, std, edges)),
        "reliability_eq20": reliability_gaussian(y, mean, std),
    }
