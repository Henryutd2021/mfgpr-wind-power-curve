import numpy as np
import pytest
from scipy.integrate import trapezoid

from mfgpr_framework.kde import kde_density, kde_sample, scott_bandwidth


def test_density_integrates_to_one_1d():
    pts = np.random.default_rng(0).normal(3, 1, 300)
    grid = np.linspace(-3, 9, 2001)
    assert trapezoid(kde_density(pts, grid, 0.3), grid) == pytest.approx(1.0, abs=1e-3)


def test_density_matches_eqs_1_to_3_in_2d():
    pts = np.array([[0.0, 0.0], [1.0, 2.0]])
    h = np.array([0.5, 2.0])
    z = np.array([[0.5, 1.0]])
    phi = lambda u: np.exp(-u**2 / 2) / np.sqrt(2 * np.pi)
    expected = np.mean([np.prod(phi((z[0] - p) / h) / h) for p in pts])
    np.testing.assert_allclose(kde_density(pts, z, h), [expected])


def test_scott_bandwidth():
    pts = np.random.default_rng(1).normal(size=(400, 2)) * [2.0, 50.0]
    h = scott_bandwidth(pts)
    np.testing.assert_allclose(h, pts.std(axis=0, ddof=1) * 400 ** (-1 / 6))


def test_samples_follow_the_density():
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.normal(8, 3, 5000), rng.normal(900, 400, 5000)])
    h = np.array([0.5, 40.0])
    s = kde_sample(pts, 20000, h, rng=3)
    assert s.shape == (20000, 2)
    np.testing.assert_allclose(s.mean(axis=0), pts.mean(axis=0), rtol=0.02)
    np.testing.assert_allclose(s.var(axis=0), pts.var(axis=0) + h**2, rtol=0.05)


def test_limits_and_errors():
    pts = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 10, 50)])
    s = kde_sample(pts, 500, [0.3, 3.0], rng=0, lower=[0, 0])
    assert np.all(s >= 0)
    with pytest.raises(ValueError):
        kde_sample(pts, 10, -1.0)
    with pytest.raises(ValueError):
        kde_sample(pts, 0)
    with pytest.raises(ValueError):
        kde_sample(np.array([[np.nan, 1.0], [0.0, 1.0]]), 5)
    with pytest.raises(RuntimeError):
        kde_sample(pts, 10, [0.01, 0.01], rng=0, lower=[5, 5], max_tries=3)
