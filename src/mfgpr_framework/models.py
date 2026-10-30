"""The three power-curve estimators compared in the paper (Section 4), behind one interface.

* ``standard_gpr``: GP regression on the high-fidelity data only (Section 2.2);
* ``linear_mf``: linear multi-fidelity GP, Eq. (6);
* ``nonlinear_mf``: nonlinear multi-fidelity GP, Eqs. (10)-(12).

Each model is fitted on a list of fidelity levels, lowest first, as (x, y) pairs: for a power curve,
level 0 holds points of the theoretical curve and the last level holds high-fidelity measured points
(wind speed in m/s, power in kW). ``predict`` returns the mean and the standard deviation in kW.

Output scaling: all outputs are divided by one common scale s (default: the standard deviation of
all training outputs pooled) before fitting, and predictions are multiplied back. The GP prior mean
stays zero (Eq. 4). Inputs are not scaled; length scales are learnt in m/s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gp import GaussianNoise, GaussianProcess
from .multifidelity import LinearMultiFidelityGP, NonlinearMultiFidelityGP, as_2d, default_rbf

KINDS = ("nonlinear_mf", "linear_mf", "standard_gpr")
LABELS = {"nonlinear_mf": "Nonlinear MF-GPR", "linear_mf": "Linear MF-GPR", "standard_gpr": "Standard GPR"}


@dataclass
class ModelOptions:
    """Fitting options shared by the three models."""

    restarts: int = 5            # random restarts of the marginal-likelihood optimisation
    n_samples: int = 100         # Monte-Carlo samples (nonlinear model only)
    noise_variance: float = 1e-2  # starting noise variance, as a fraction of var(y)
    output_scale: float | None = None  # None: pooled standard deviation of the training outputs
    seed: int | None = 0


class PowerCurveModel:
    """Common wrapper: scaling, fitting on fidelity levels, prediction in physical units."""

    kind = ""

    def __init__(self, options: ModelOptions | None = None):
        self.options = options or ModelOptions()
        self.scale = 1.0
        self._model = None

    def _set_scale(self, levels):
        if self.options.output_scale is not None:
            if self.options.output_scale <= 0:
                raise ValueError("output_scale must be positive")
            self.scale = float(self.options.output_scale)
        else:
            pooled = np.concatenate([np.asarray(y, dtype=float).ravel() for _, y in levels])
            sd = float(np.std(pooled))
            self.scale = sd if sd > 0 else 1.0

    def fit(self, levels):
        levels = [(as_2d(x), np.asarray(y, dtype=float).ravel()) for x, y in levels]
        self._set_scale(levels)
        scaled = [(x, y / self.scale) for x, y in levels]
        self._fit_scaled(scaled)
        return self

    def _fit_scaled(self, levels):  # pragma: no cover - abstract
        raise NotImplementedError

    def _predict_scaled(self, X, include_noise):  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, x, include_noise: bool = True):
        """Mean and standard deviation (kW) of the high-fidelity output at inputs ``x``.

        ``include_noise=True`` (default) gives the predictive distribution of a measured value;
        ``False`` gives the distribution of the latent power curve f(x) (Eq. 5). For the nonlinear
        model these are the first two moments of a Monte-Carlo mixture (see ``predict_components``).
        """
        if self._model is None:
            raise RuntimeError("fit the model before predicting")
        mean, var = self._predict_scaled(as_2d(x), include_noise)
        return mean * self.scale, np.sqrt(var) * self.scale


class StandardGPR(PowerCurveModel):
    kind = "standard_gpr"

    def _fit_scaled(self, levels):
        X, y = levels[-1]  # high-fidelity data only
        noise = GaussianNoise(self.options.noise_variance * max(np.var(y), 1e-12))
        self._model = GaussianProcess(default_rbf(X, y), noise)
        self._model.fit(X, y, restarts=self.options.restarts, seed=self.options.seed)

    def _predict_scaled(self, X, include_noise):
        return self._model.predict(X, include_noise=include_noise)


class LinearMFGPR(PowerCurveModel):
    kind = "linear_mf"

    def _fit_scaled(self, levels):
        self._model = LinearMultiFidelityGP(noise_variance=self.options.noise_variance)
        self._model.fit(levels, restarts=self.options.restarts, seed=self.options.seed)

    def _predict_scaled(self, X, include_noise):
        return self._model.predict(X, include_noise=include_noise)


class NonlinearMFGPR(PowerCurveModel):
    kind = "nonlinear_mf"

    def _fit_scaled(self, levels):
        self._model = NonlinearMultiFidelityGP(n_samples=self.options.n_samples,
                                               noise_variance=self.options.noise_variance)
        self._model.fit(levels, restarts=self.options.restarts, seed=self.options.seed)

    def _predict_scaled(self, X, include_noise):
        return self._model.predict(X, include_noise=include_noise)

    def predict_components(self, x, include_noise: bool = True):
        """Monte-Carlo mixture components (kW): means and standard deviations, both S x n.

        The predictive distribution is the equal-weight mixture of the S Gaussians N(means[s], stds[s]**2);
        ``include_noise`` adds the high-fidelity noise variance to every component.
        """
        if self._model is None:
            raise RuntimeError("fit the model before predicting")
        means, variances = self._model.predict_components(as_2d(x))
        if include_noise:
            variances = variances + self._model.gps[-1].noise.variance
        return means * self.scale, np.sqrt(variances) * self.scale


def make_model(kind: str, options: ModelOptions | None = None) -> PowerCurveModel:
    """Create an unfitted model of one of ``KINDS``."""
    classes = {"standard_gpr": StandardGPR, "linear_mf": LinearMFGPR, "nonlinear_mf": NonlinearMFGPR}
    if kind not in classes:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {KINDS}")
    return classes[kind](options)
