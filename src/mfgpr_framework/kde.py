"""Kernel density estimation (Section 2.1, Eqs. 1-3) and sampling of high-fidelity training points.

Eq. (1) estimates a density by placing a Gaussian kernel on every record. For (wind speed, power)
records the framework uses the Gaussian kernel density estimate with one bandwidth per coordinate,
h_k being the standard deviation of the Gaussian kernel in coordinate k:

    f_h(z) = 1/n * sum_i prod_k (1/h_k) * phi((z_k - z_ik) / h_k),

with phi the standard normal density (Eq. 3). Drawing from this density is a "smoothed bootstrap":
pick a record uniformly at random and add Gaussian noise with standard deviation h_k to coordinate k.
The draws follow the density of the data, so a few hundred points keep the shape of the full
scatter, including its noisy parts, without any cleaning.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _as_points(points) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    p = p.reshape(-1, 1) if p.ndim == 1 else p
    if p.shape[0] < 2:
        raise ValueError("need at least two records")
    if not np.all(np.isfinite(p)):
        raise ValueError("records must be finite")
    return p


def _bandwidth(points: np.ndarray, bandwidth) -> np.ndarray:
    if bandwidth is None:
        return scott_bandwidth(points)
    h = np.broadcast_to(np.asarray(bandwidth, dtype=float), (points.shape[1],)).copy()
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        raise ValueError("bandwidths must be positive and finite")
    return h


def scott_bandwidth(points) -> np.ndarray:
    """Scott's rule per coordinate: h_k = sigma_k * n**(-1/(d+4))."""
    p = _as_points(points)
    n, d = p.shape
    sd = p.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return sd * n ** (-1.0 / (d + 4))


def kde_density(points, z, bandwidth=None) -> np.ndarray:
    """Kernel density estimate (formula in the module docstring) at the rows of ``z``."""
    p = _as_points(points)
    z = np.asarray(z, dtype=float)
    z = z.reshape(-1, p.shape[1])
    h = _bandwidth(p, bandwidth)
    dens = np.zeros(len(z))
    for start in range(0, len(z), 2048):  # chunked to bound memory
        u = (z[start:start + 2048, None, :] - p[None, :, :]) / h
        dens[start:start + 2048] = (norm.pdf(u) / h).prod(axis=2).mean(axis=1)
    return dens


def kde_sample(points, n: int, bandwidth=None, rng=None, lower=None, upper=None, max_tries: int = 100) -> np.ndarray:
    """Draw ``n`` points from the kernel density estimate of ``points`` (rows = records).

    ``lower``/``upper`` (scalars or one value per coordinate) reject draws outside physical limits,
    for example negative wind speed or power; rejected draws are redrawn.
    """
    if n < 1:
        raise ValueError("n must be positive")
    p = _as_points(points)
    h = _bandwidth(p, bandwidth)
    rng = np.random.default_rng(rng)
    lo = np.full(p.shape[1], -np.inf) if lower is None else np.broadcast_to(np.asarray(lower, float), (p.shape[1],))
    hi = np.full(p.shape[1], np.inf) if upper is None else np.broadcast_to(np.asarray(upper, float), (p.shape[1],))
    out = np.empty((0, p.shape[1]))
    for _ in range(max_tries):
        need = n - len(out)
        draws = p[rng.integers(len(p), size=need)] + rng.standard_normal((need, p.shape[1])) * h
        keep = np.all((draws >= lo) & (draws <= hi), axis=1)
        out = np.vstack([out, draws[keep]])
        if len(out) >= n:
            return out[:n]
    raise RuntimeError("too many draws fell outside the limits; check lower/upper and the bandwidth")
