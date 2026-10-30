"""Synthetic example data: an invented turbine type and an invented wind farm.

Nothing here describes a real turbine or wind farm. The data only show how the framework behaves:
the "measured" records deviate from the theoretical curve (each turbine has its own shifted, slightly
derated, smoother curve), the scatter is widest on the ramp, and a share of records shows stops
(power near zero) and curtailment (power capped), as in real operating data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def synthetic_power_curve(rated_kw: float = 2000.0, cut_in: float = 3.0, rated_speed: float = 12.0,
                          cut_out: float = 25.0, step: float = 0.25, max_speed: float = 28.0) -> pd.DataFrame:
    """Theoretical curve: cubic between cut-in and rated speed, flat to cut-out, zero elsewhere."""
    v = np.round(np.arange(0.0, max_speed + step / 2, step), 6)
    p = np.where((v >= cut_in) & (v < rated_speed),
                 rated_kw * (v**3 - cut_in**3) / (rated_speed**3 - cut_in**3), 0.0)
    p = np.where((v >= rated_speed) & (v <= cut_out), rated_kw, p)
    return pd.DataFrame({"wind_speed": v, "power": p})


def _wind_series(n: int, rng, phi: float, common: np.ndarray, weibull_k: float, weibull_c: float):
    own = np.empty(n)
    own[0] = rng.standard_normal()
    for t in range(1, n):
        own[t] = phi * own[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    z = 0.8 * common + 0.6 * own  # unit variance
    u = np.clip(norm.cdf(z), 1e-9, 1 - 1e-9)
    return weibull_c * (-np.log(1 - u)) ** (1 / weibull_k)


def synthetic_farm(n_turbines: int = 6, days: int = 4, seed: int = 0, rated_kw: float = 2000.0,
                   cut_out: float = 25.0, interval_minutes: int = 10, weibull_k: float = 2.0,
                   weibull_c: float = 8.0, stop_share: float = 0.04, curtail_share: float = 0.05) -> pd.DataFrame:
    """Records of ``n_turbines`` turbines over ``days`` days (records schema of ``data.py``)."""
    if n_turbines < 1 or days < 1:
        raise ValueError("n_turbines and days must be positive")
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / interval_minutes)
    phi = 0.97
    common = np.empty(n)
    common[0] = rng.standard_normal()
    for t in range(1, n):
        common[t] = phi * common[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal()
    times = pd.to_timedelta(np.arange(n) * interval_minutes, unit="min").astype(str)
    frames = []
    for k in range(n_turbines):
        v = _wind_series(n, rng, phi, common, weibull_k, weibull_c)
        mid = 8.0 + rng.uniform(-0.6, 0.6)      # speed at half of rated power
        width = 1.1 + rng.uniform(-0.15, 0.15)  # smoothness of the ramp
        derate = rng.uniform(0.93, 1.0)
        frac = 1.0 / (1.0 + np.exp(-(v - mid) / width))
        frac = np.where(v > cut_out, 0.0, frac)
        sd = rated_kw * (0.015 + 0.10 * frac * (1 - frac))
        p = derate * rated_kw * frac + sd * rng.standard_normal(n)
        stop = rng.random(n) < stop_share
        p[stop] = rng.uniform(-5.0, 15.0, stop.sum())
        cap = rated_kw * rng.uniform(0.35, 0.7)
        curtail = rng.random(n) < curtail_share
        p[curtail] = np.minimum(p[curtail], cap + 10.0 * rng.standard_normal(curtail.sum()))
        p = np.clip(p, -10.0, 1.02 * rated_kw)
        frames.append(pd.DataFrame({"turbine_id": k + 1, "time": times, "wind_speed": np.round(v, 2),
                                    "power": np.round(p, 2)}))
    return pd.concat(frames, ignore_index=True)
