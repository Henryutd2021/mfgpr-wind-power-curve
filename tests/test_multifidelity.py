import numpy as np
import pytest

from mfgpr_framework.gp import GaussianProcess, LevelNoise
from mfgpr_framework.kernels import RBF
from mfgpr_framework.models import LinearMFGPR, ModelOptions, NonlinearMFGPR, StandardGPR, make_model
from mfgpr_framework.multifidelity import (LinearMultiFidelityGP, LinearMultiFidelityKernel,
                                           NonlinearMultiFidelityGP)

from conftest import numeric_gradient


def test_linear_kernel_structure_two_levels():
    k0, k1 = RBF([0], 1.3, 0.8), RBF([0], 0.4, 1.2)
    k = LinearMultiFidelityKernel([k0, k1], rho=[1.7])
    x = np.array([[0.2], [1.1]])
    X0, X1 = np.hstack([x, [[0], [0]]]), np.hstack([x, [[1], [1]]])
    np.testing.assert_allclose(k.K(X0, X0), k0.K(x))
    np.testing.assert_allclose(k.K(X1, X0), 1.7 * k0.K(x))
    np.testing.assert_allclose(k.K(X1, X1), 1.7**2 * k0.K(x) + k1.K(x))
    np.testing.assert_allclose(k.Kdiag(X1), np.diag(k.K(X1)))


def test_linear_kernel_three_levels_and_gradients():
    k = LinearMultiFidelityKernel([RBF([0], 1.0, 1.0), RBF([0], 0.5, 0.8), RBF([0], 0.2, 1.3)], rho=[1.3, 0.8])
    rng = np.random.default_rng(0)
    X = np.hstack([rng.uniform(0, 3, (15, 1)), rng.integers(0, 3, (15, 1))])
    x = np.array([[0.4]])
    a = lambda t: np.hstack([x, [[t]]])
    expected = 1.3 * 0.8 * k.base[0].K(x)  # cov(f2, f0) = rho0 * rho1 * c0
    np.testing.assert_allclose(k.K(a(2), a(0)), expected)
    gp = GaussianProcess(k, LevelNoise(3, [0.01, 0.02, 0.03]))
    gp.X, gp.y = X, np.sin(X[:, 0]) * (1 + X[:, 1])
    theta = gp.theta
    numeric = numeric_gradient(lambda t: gp.neg_log_marginal_likelihood(t, gradient=False), theta)
    np.testing.assert_allclose(gp.neg_log_marginal_likelihood(theta)[1], numeric, rtol=1e-4, atol=1e-4)


def test_linear_model_recovers_scaling_factor():
    xl = np.linspace(0, 1, 30)
    xh = np.linspace(0, 1, 8)
    low = np.sin(6 * xl)
    high = 2.0 * np.sin(6 * xh) + 0.3 * xh
    model = LinearMultiFidelityGP().fit([(xl, low), (xh, high)], restarts=3, seed=0)
    assert abs(model.rho[0] - 2.0) < 0.3
    mean, var = model.predict(np.linspace(0, 1, 50))
    assert np.sqrt(np.mean((mean - (2 * np.sin(6 * np.linspace(0, 1, 50)) + 0.3 * np.linspace(0, 1, 50))) ** 2)) < 0.1
    assert np.all(var >= 0)


def _benchmark():
    """Nonlinear two-level test function of Perdikaris et al. (2017)."""
    f_low = lambda x: np.sin(8 * np.pi * x)
    f_high = lambda x: (x - np.sqrt(2)) * f_low(x) ** 2
    xl = np.linspace(0, 1, 50)
    xh = np.linspace(0, 1, 14)
    return [(xl, f_low(xl)), (xh, f_high(xh))], f_high


def test_nonlinear_model_beats_single_fidelity_on_benchmark():
    levels, f_high = _benchmark()
    xs = np.linspace(0, 1, 200)
    err = {}
    for kind in ("nonlinear_mf", "linear_mf", "standard_gpr"):
        model = make_model(kind, ModelOptions(restarts=3, n_samples=50, seed=0)).fit(levels)
        mean, _ = model.predict(xs, include_noise=False)
        err[kind] = np.sqrt(np.mean((mean - f_high(xs)) ** 2))
    assert err["nonlinear_mf"] < 0.5 * err["standard_gpr"]
    assert err["nonlinear_mf"] < err["linear_mf"]


def test_nonlinear_prediction_is_deterministic_and_levels_work():
    levels, _ = _benchmark()
    xm = np.linspace(0, 1, 20)
    three = levels + [(levels[1][0][::2], 1.5 * levels[1][1][::2])]
    model = NonlinearMultiFidelityGP(n_samples=20).fit(three, restarts=0, seed=1)
    m1, v1 = model.predict(xm)
    m2, v2 = model.predict(xm)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(v1, v2)
    for level in range(3):
        mean, var = model.predict(xm, level=level)
        assert mean.shape == var.shape == (20,) and np.all(var >= 0)
    _, v_noisy = model.predict(xm, include_noise=True)
    assert np.all(v_noisy >= v1)
    with pytest.raises(ValueError):
        model.predict(xm, level=3)


def test_level_checks():
    with pytest.raises(ValueError):
        LinearMultiFidelityGP().fit([(np.zeros(3), np.zeros(3))])
    with pytest.raises(ValueError):
        NonlinearMultiFidelityGP().fit([(np.zeros(3), np.zeros(2)), (np.zeros(3), np.zeros(3))])
    with pytest.raises(ValueError):
        NonlinearMultiFidelityGP(n_samples=1)
    with pytest.raises(ValueError):
        make_model("unknown")


def test_wrappers_scale_outputs(small_options):
    levels, _ = _benchmark()
    big = [(x, 1000 * y) for x, y in levels]
    for cls in (StandardGPR, LinearMFGPR, NonlinearMFGPR):
        a = cls(small_options).fit(levels)
        b = cls(small_options).fit(big)
        assert b.scale == pytest.approx(1000 * a.scale)
        ma, sa = a.predict([0.3, 0.6])
        mb, sb = b.predict([0.3, 0.6])
        # same scaled problem; the optimiser may stop at slightly different points (round-off)
        np.testing.assert_allclose(mb, 1000 * ma, rtol=5e-3, atol=1e-3)
        np.testing.assert_allclose(sb, 1000 * sa, rtol=5e-3, atol=1e-3)
    with pytest.raises(RuntimeError):
        StandardGPR().predict([1.0])
    with pytest.raises(ValueError):
        StandardGPR(ModelOptions(output_scale=-1)).fit(levels)


def test_prediction_input_and_level_checks():
    rng = np.random.default_rng(0)
    X0, X1 = rng.uniform(0, 1, (20, 2)), rng.uniform(0, 1, (8, 2))
    levels = [(X0, X0.sum(axis=1)), (X1, 2 * X1.sum(axis=1))]
    for model in (LinearMultiFidelityGP().fit(levels, restarts=0), NonlinearMultiFidelityGP(n_samples=5).fit(levels, restarts=0)):
        with pytest.raises(ValueError):
            model.predict([[0.3]])            # one column instead of two
        for bad in (-1, 2, 0.9, True):
            with pytest.raises(ValueError):
                model.predict(X1, level=bad)
        mean, var = model.predict(X1, level=np.int64(0))
        assert mean.shape == (8,)


def test_mixture_components_match_moments():
    levels, _ = _benchmark()
    model = NonlinearMFGPR(ModelOptions(restarts=1, n_samples=30, seed=0)).fit(levels)
    xs = np.linspace(0, 1, 15)
    means, stds = model.predict_components(xs)
    assert means.shape == stds.shape == (30, 15)
    mean, std = model.predict(xs)
    np.testing.assert_allclose(means.mean(axis=0), mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.sqrt((stds**2).mean(axis=0) + means.var(axis=0)), std, rtol=1e-10)
