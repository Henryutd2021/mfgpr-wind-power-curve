"""Command line: ``mfgpr-framework synthetic DIR`` and ``mfgpr-framework run DIR --out-dir OUT``."""
from __future__ import annotations

import argparse
import sys

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
    r.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "synthetic":
        d = write_inputs(args.directory, synthetic_farm(args.turbines, args.days, seed=args.seed),
                         synthetic_power_curve())
        print(f"wrote {d}/records.csv and {d}/power_curve.csv (synthetic)")
        return 0
    records, curve = read_inputs(args.directory)
    config = ExperimentConfig(n_high=args.n_high, n_low=args.n_low, models=tuple(args.models),
                              n_turbines=args.n_turbines, seed=args.seed,
                              model_options=ModelOptions(restarts=args.restarts, n_samples=args.n_samples,
                                                         seed=args.seed))
    result = run_experiment(records, curve, config)
    out = result.save(args.out_dir)
    print(result.summary().to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"tables written to {out}")
    return 0
