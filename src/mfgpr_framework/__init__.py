"""Multi-fidelity Gaussian process regression for probabilistic wind power curves (framework).

Framework of: H. Li, C. Feng, J. Zhang, "A Multi-Fidelity Gaussian Process Regression Method for
Probabilistic Wind Farm Power Curve Estimation", ASME IDETC/CIE 2023, doi:10.1115/DETC2023-114762.
"""
from .data import (DataError, from_sdwpf, low_fidelity_points, prepare_records, read_inputs,
                   validate_power_curve, validate_records, write_inputs)
from .experiment import ExperimentConfig, ExperimentResult, run_experiment
from .gp import GaussianNoise, GaussianProcess, LevelNoise
from .kde import kde_density, kde_sample, scott_bandwidth
from .kernels import RBF, Product, Sum
from .models import KINDS, LABELS, LinearMFGPR, ModelOptions, NonlinearMFGPR, StandardGPR, make_model
from .multifidelity import LinearMultiFidelityGP, LinearMultiFidelityKernel, NonlinearMultiFidelityGP
from .synthetic import synthetic_farm, synthetic_power_curve

__version__ = "1.0.1"
__all__ = [
    "DataError", "from_sdwpf", "low_fidelity_points", "prepare_records", "read_inputs",
    "validate_power_curve", "validate_records", "write_inputs",
    "ExperimentConfig", "ExperimentResult", "run_experiment",
    "GaussianNoise", "GaussianProcess", "LevelNoise",
    "kde_density", "kde_sample", "scott_bandwidth",
    "RBF", "Product", "Sum",
    "KINDS", "LABELS", "LinearMFGPR", "ModelOptions", "NonlinearMFGPR", "StandardGPR", "make_model",
    "LinearMultiFidelityGP", "LinearMultiFidelityKernel", "NonlinearMultiFidelityGP",
    "synthetic_farm", "synthetic_power_curve",
]
