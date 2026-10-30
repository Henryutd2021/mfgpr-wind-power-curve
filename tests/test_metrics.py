import numpy as np
import pytest
from scipy.stats import norm

from mfgpr_framework import metrics as mt


def test_deterministic_metrics_by_hand():
    a = np.array([100.0, 0.0, 300.0, 400.0])
    p = np.array([110.0, 20.0, 270.0, 400.0])
    d = mt.deterministic_metrics(a, p, x_max=500.0)
    assert d["rmse"] == pytest.approx(np.sqrt((100 + 400 + 900) / 4))
    assert d["mae"] == pytest.approx(60 / 4)
    assert d["nrmse"] == pytest.approx(d["rmse"] / 500)
    assert d["nmae"] == pytest.approx(d["mae"] / 500)
    assert d["mape_percent"] == pytest.approx(100 * (0.1 + 0.1 + 0.0) / 3)  # x = 0 excluded
    assert mt.mape([0.0, 0.0], [1.0, 2.0]) != mt.mape([0.0, 0.0], [1.0, 2.0])  # nan
    assert mt.mape(a, p, min_actual=150) == pytest.approx(100 * 0.1 / 2)
    with pytest.raises(ValueError):
        mt.nrmse(a, p, 0.0)
    with pytest.raises(ValueError):
        mt.rmse(a, p[:2])


def test_pinball_loss():
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 2.0, 2.0])
    assert mt.pinball_loss(y, q, 0.5) == pytest.approx(mt.mae(y, q) / 2)
    assert mt.pinball_loss(y, q, 0.9) == pytest.approx((0.1 * 1 + 0 + 0.9 * 1) / 3)


def test_reliability_closed_form_matches_numeric_integral():
    rng = np.random.default_rng(0)
    y, mean, std = rng.normal(size=40), rng.normal(size=40), rng.uniform(0.5, 2, 40)
    grid = np.linspace(-15, 15, 60001)  # the indicator jumps, so the trapezoidal error is O(grid step)
    numeric = mt.reliability_numeric(y, lambda z: norm.cdf((z[None, :] - mean[:, None]) / std[:, None]), grid)
    assert mt.reliability_gaussian(y, mean, std) == pytest.approx(numeric, rel=1e-4)


def test_bin_probabilities_and_sharpness_eq19():
    edges = np.linspace(0, 1000, 11)
    p = mt.bin_probabilities([100.0, 500.0, 900.0], [50.0, 50.0, 50.0], edges)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert mt.sharpness(np.tile(p[0], (5, 1))) == pytest.approx(0.0)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert mt.sharpness(q) == pytest.approx((0.25 * 4) / 2)
    with pytest.raises(ValueError):
        mt.bin_probabilities([0.0], [1.0], [0.0, 1.0])


def test_interval_curves_for_calibrated_forecasts():
    rng = np.random.default_rng(1)
    mean = rng.normal(size=20000)
    std = np.full(20000, 2.0)
    y = mean + std * rng.standard_normal(20000)
    for row in mt.interval_curves(y, mean, std, [0.2, 0.5, 0.9]):
        assert row["coverage"] == pytest.approx(row["nominal"], abs=0.01)
        assert row["mean_width"] == pytest.approx(2 * norm.ppf(0.5 + row["nominal"] / 2) * 2.0)
    with pytest.raises(ValueError):
        mt.interval_coverage(y, mean, -std, 0.5)


def test_domain_checks():
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        mt.pinball_loss(y, y, -1.0)
    with pytest.raises(ValueError):
        mt.gaussian_quantile(y, [1.0, 1.0], 1.0)
    with pytest.raises(ValueError):
        mt.bin_probabilities(y, [1.0, -1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        mt.interval_width([1.0, 1.0], -0.5)
    with pytest.raises(ValueError):
        mt.interval_coverage(y, y, [1.0, 1.0], 1.5)
    with pytest.raises(ValueError):
        mt.reliability_gaussian([np.nan, 1.0], y, [1.0, 1.0])
    with pytest.raises(ValueError):
        mt.sharpness([[1.5, -0.5]])


def test_mixture_cdf():
    z = np.linspace(-3, 3, 7)
    single = mt.mixture_cdf(z, [[0.5, -1.0]], [[1.0, 2.0]])
    np.testing.assert_allclose(single, norm.cdf((z[None, :] - np.array([[0.5], [-1.0]])) / np.array([[1.0], [2.0]])))
    two = mt.mixture_cdf(z, [[0.0], [2.0]], [[1.0], [1.0]])
    np.testing.assert_allclose(two[0], 0.5 * (norm.cdf(z) + norm.cdf(z - 2.0)))
    with pytest.raises(ValueError):
        mt.mixture_cdf(z, [[0.0]], [[0.0]])


def test_quantile_shape_and_non_finite_inputs():
    q = mt.gaussian_quantile(np.zeros((2, 1)), np.ones((2, 1)), 0.9)
    assert q.shape == (2, 1)
    with pytest.raises(ValueError):
        mt.gaussian_quantile([np.nan, 0.0], [1.0, 1.0], 0.5)
    with pytest.raises(ValueError):
        mt.pinball_loss([np.nan, 1.0], [0.0, 0.0], 0.5)
    with pytest.raises(ValueError):
        mt.bin_probabilities([0.0], [1.0], [0.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        mt.rmse([1.0, np.inf], [1.0, 1.0])


def test_scalar_broadcasting_and_infinite_outer_edges():
    np.testing.assert_allclose(mt.gaussian_quantile([0.0, 1.0], 1.0, 0.5), [0.0, 1.0])
    with pytest.raises(ValueError):
        mt.gaussian_quantile([0.0, 1.0], [1.0, 1.0, 1.0], 0.5)
    np.testing.assert_allclose(mt.bin_probabilities([0.0], [1.0], [-np.inf, 0.0, np.inf]), [[0.5, 0.5]])
    with pytest.raises(ValueError):
        mt.bin_probabilities([0.0], [1.0], [0.0, np.inf, 5.0])
