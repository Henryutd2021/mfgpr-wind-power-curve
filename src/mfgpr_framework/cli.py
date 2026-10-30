"""Command line: ``mfgpr-framework synthetic DIR`` and ``mfgpr-framework run DIR --out-dir OUT``."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .data import read_inputs, write_inputs
from .experiment import ExperimentConfig, run_experiment
from .models import KINDS, ModelOptions
from .synthetic import synthetic_farm, synthetic_power_curve


def _parser():
    p = argparse.ArgumentParser(prog="mfgpr-framework",
                                description="Multi-fidelity GP power-curve estimation (framework).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="write a synthetic input directory")
    s.add_argument("directory")
    s.add_argument("--turbines", type=int, default=6)
    s.add_argument("--days", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="run the experiment design on an input directory")
    r.add_argument("directory")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--n-high", type=int, default=200)
    r.add_argument("--n-low", type=int, default=200)
    r.add_argument("--models", nargs="+", choices=KINDS, default=list(KINDS))
    r.add_argument("--n-turbines", type=int, default=None, help="random selection; default: all turbines")
    r.add_argument("--restarts", type=int, default=5)
    r.add_argument("--n-samples", type=int, default=100, help="Monte-Carlo samples of the nonlinear model")
    t = r.add_mutually_exclusive_group()
    t.add_argument("--mape-min-fraction", type=float, default=0.05, metavar="F",
                   help="Eq. 17 leaves out records with |power| <= F * x_max, where x_max is the rated power "
                        "(maximum of the theoretical curve, the normalizer of NRMSE and NMAE); default 0.05")
    t.add_argument("--mape-min-actual", type=float, default=None, metavar="KW",
                   help="absolute threshold in kW instead of --mape-min-fraction "
                        "(0 leaves out only records with zero power)")
    r.add_argument("--seed", type=int, default=0)
    return p


def _mape_threshold(args, curve):
    """Threshold of Eq. 17 in kW and a description of how it was set."""
    if args.mape_min_actual is not None:
        if not (np.isfinite(args.mape_min_actual) and args.mape_min_actual >= 0):
            raise SystemExit("--mape-min-actual must be finite and >= 0")
        return float(args.mape_min_actual), "--mape-min-actual"
    f = args.mape_min_fraction
    if not (np.isfinite(f) and 0 <= f < 1):
        raise SystemExit("--mape-min-fraction must be in [0, 1)")
    x_max = float(curve["power"].max())
    return f * x_max, f"{100 * f:g} % of the rated power, {x_max:.6g} kW"


def main(argv=None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "synthetic":
        d = write_inputs(args.directory, synthetic_farm(args.turbines, args.days, seed=args.seed),
                         synthetic_power_curve())
        print(f"wrote {d}/records.csv and {d}/power_curve.csv (synthetic)")
        return 0
    records, curve = read_inputs(args.directory)
    min_actual, rule = _mape_threshold(args, curve)
    config = ExperimentConfig(n_high=args.n_high, n_low=args.n_low, models=tuple(args.models),
                              n_turbines=args.n_turbines, seed=args.seed, mape_min_actual=min_actual,
                              model_options=ModelOptions(restarts=args.restarts, n_samples=args.n_samples,
                                                         seed=args.seed))
    result = run_experiment(records, curve, config)
    out = result.save(args.out_dir)
    print(f"Deterministic metrics, mean over turbines; MAPE leaves out records with |power| <= "
          f"{min_actual:.6g} kW ({rule}):")
    print(result.summary().to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"tables written to {out}")
    return 0
