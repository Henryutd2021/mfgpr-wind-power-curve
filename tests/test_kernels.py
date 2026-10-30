import numpy as np
import pytest

from mfgpr_framework.kernels import RBF


def _kernel_grad_check(kernel, X):
    theta0 = kernel.theta.copy()
    analytic = kernel.gradients(X)
    for i in range(len(theta0)):
        step = np.zeros_like(theta0)
        step[i] = 1e-6
        kernel.theta = theta0 + step
        Kp = kernel.K(X)
        kernel.theta = theta0 - step
        Km = kernel.K(X)
        kernel.theta = theta0
        np.testing.assert_allclose(analytic[i], (Kp - Km) / 2e-6, atol=1e-6)


def test_rbf_matches_eq12():
    X = np.array([[0.0, 1.0], [2.0, -1.0], [1.5, 0.5]])
    k = RBF([0, 1], variance=2.0, lengthscale=[0.5, 2.0])
    w = 1 / np.array([0.5, 2.0]) ** 2  # ARD weights of Eq. (12)
    expected = np.array([[2.0 * np.exp(-0.5 * np.sum(w * (a - b) ** 2)) for b in X] for a in X])
    np.testing.assert_allclose(k.K(X), expected)
    np.testing.assert_allclose(k.Kdiag(X), np.diag(expected))


def test_active_dims_ignore_other_columns():
    X = np.array([[0.0, 5.0], [1.0, -3.0]])
    k = RBF([0], 1.0, 1.0)
    np.testing.assert_allclose(k.K(X)[0, 1], np.exp(-0.5))


@pytest.mark.parametrize("make", [
    lambda: RBF([0], 1.3, 0.7),
    lambda: RBF([0, 1], 0.8, [0.6, 1.4]),
    lambda: RBF([1], 1.0, 0.9, fixed_variance=True),
    lambda: RBF([0], 1.2, 0.7) * RBF([1], 1.0, 0.8, fixed_variance=True) + RBF([0], 0.3, 1.1),
])
def test_gradients_match_finite_differences(make):
    X = np.random.default_rng(0).uniform(0, 3, (12, 2))
    _kernel_grad_check(make(), X)


def test_composite_kernel_is_psd_and_theta_round_trips():
    X = np.random.default_rng(1).uniform(0, 3, (20, 2))
    k = RBF([0], 1.2, 0.7) * RBF([1], 1.0, 0.8, fixed_variance=True) + RBF([0], 0.3, 1.1)
    assert np.min(np.linalg.eigvalsh(k.K(X))) > -1e-10
    theta = k.theta + 0.1
    k.theta = theta
    np.testing.assert_allclose(k.theta, theta)
    assert len(k.param_names) == len(theta) == len(k.bounds())


def test_invalid_parameters():
    with pytest.raises(ValueError):
        RBF([0], variance=-1.0)
    with pytest.raises(ValueError):
        RBF([], 1.0)
