import numpy as np
import pytest

from mfgpr_framework.gp import GaussianNoise, GaussianProcess, LevelNoise
from mfgpr_framework.kernels import RBF

from conftest import numeric_gradient


def _data(n=25, seed=0):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 6, n)).reshape(-1, 1)
    return X, np.sin(X[:, 0]) + 0.05 * rng.standard_normal(n)


def test_posterior_matches_eqs_4_and_5():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.1, 0.9), GaussianNoise(0.01)).fit(X, y, optimize=False)
    Xs = np.linspace(0, 6, 7).reshape(-1, 1)
    C = gp.kernel.K(X) + (0.01 + gp.jitter) * np.eye(len(X))
    cs = gp.kernel.K(Xs, X)
    mean, var = gp.predict(Xs)
    np.testing.assert_allclose(mean, cs @ np.linalg.solve(C, y), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(var, 1.1 - np.einsum("ij,ji->i", cs, np.linalg.solve(C, cs.T)), atol=1e-10)
    _, var_y = gp.predict(Xs, include_noise=True)
    np.testing.assert_allclose(var_y - var, 0.01)


def test_marginal_likelihood_gradient():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.1, 0.9), GaussianNoise(0.02))
    gp.X, gp.y = X, y
    theta = gp.theta
    analytic = gp.neg_log_marginal_likelihood(theta)[1]
    numeric = numeric_gradient(lambda t: gp.neg_log_marginal_likelihood(t, gradient=False), theta)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_fit_recovers_smooth_function():
    X, y = _data(40)
    gp = GaussianProcess(RBF([0], 1.0, 1.0), GaussianNoise(0.1)).fit(X, y, restarts=2, seed=0)
    Xs = np.linspace(0.5, 5.5, 50).reshape(-1, 1)
    mean, _ = gp.predict(Xs)
    assert np.sqrt(np.mean((mean - np.sin(Xs[:, 0])) ** 2)) < 0.05
    assert gp.noise.variance < 0.01


def test_level_noise_gradients():
    X = np.hstack([np.linspace(0, 1, 6).reshape(-1, 1), np.array([[0], [0], [0], [1], [1], [1]])])
    noise = LevelNoise(2, [0.1, 0.2])
    np.testing.assert_allclose(noise.diag(X), [0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    g = noise.gradients(X)
    np.testing.assert_allclose(g[0] + g[1], noise.diag(X))


def test_errors():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.0, 1.0))
    with pytest.raises(RuntimeError):
        gp.predict(X)
    with pytest.raises(ValueError):
        gp.fit(X, np.r_[y[:-1], np.nan])
    with pytest.raises(ValueError):
        gp.fit(X, y[:-1])


def test_trial_likelihood_leaves_fitted_model_unchanged():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.0, 1.0), GaussianNoise(0.1)).fit(X, y, restarts=1, seed=0)
    before = gp.predict(X[:5])
    theta = gp.theta.copy()
    gp.neg_log_marginal_likelihood(theta + 2.0)
    np.testing.assert_array_equal(gp.theta, theta)
    for a, b in zip(before, gp.predict(X[:5])):
        np.testing.assert_allclose(a, b)


def test_changed_parameters_are_refactorised_before_predicting():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.0, 1.0), GaussianNoise(0.1)).fit(X, y, optimize=False)
    gp.kernel.theta = gp.kernel.theta + 0.5
    reference = GaussianProcess(RBF([0], np.exp(0.5), np.exp(0.5)), GaussianNoise(0.1)).fit(X, y, optimize=False)
    for a, b in zip(gp.predict(X[:4]), reference.predict(X[:4])):
        np.testing.assert_allclose(a, b, rtol=1e-10)


def test_optimum_respects_bounds_for_tiny_outputs():
    X, _ = _data()
    y = 1e-9 * np.sin(X[:, 0])
    gp = GaussianProcess(RBF([0], 1.0, 1.0), GaussianNoise(1.0)).fit(X, y, restarts=1, seed=0)
    for t, (lo, hi) in zip(gp.theta, gp.bounds()):
        assert lo - 1e-12 <= t <= hi + 1e-12


def test_prediction_input_checks():
    X, y = _data()
    gp = GaussianProcess(RBF([0], 1.0, 1.0)).fit(X, y, optimize=False)
    with pytest.raises(ValueError):
        gp.predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        gp.predict(np.array([[np.nan]]))
