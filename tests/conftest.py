import numpy as np
import pytest
from scipy.optimize import approx_fprime

import mfgpr_framework as m


@pytest.fixture(scope="session")
def farm():
    return m.synthetic_farm(n_turbines=3, days=2, seed=3)


@pytest.fixture(scope="session")
def curve():
    return m.synthetic_power_curve()


@pytest.fixture
def small_options():
    return m.ModelOptions(restarts=1, n_samples=20, seed=0)


def numeric_gradient(fun, theta, eps=1e-6):
    return approx_fprime(np.asarray(theta, dtype=float), fun, eps)
