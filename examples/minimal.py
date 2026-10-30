"""Minimal example: the experiment design on a synthetic wind farm (a few seconds).

    python examples/minimal.py            # 6 turbines, 4 days, 200 + 200 training points
    python examples/minimal.py --quick    # smaller settings (used by the tests)

The data are invented (see mfgpr_framework.synthetic); the numbers say nothing about real turbines.
"""
import argparse

import mfgpr_framework as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    turbines, days, n_high, n_low = (3, 1, 60, 40) if args.quick else (6, 4, 200, 200)
    records = m.synthetic_farm(n_turbines=turbines, days=days, seed=0)
    curve = m.synthetic_power_curve()
    config = m.ExperimentConfig(
        n_high=n_high, n_low=n_low, mape_min_actual=0.05 * curve["power"].max(),
        model_options=m.ModelOptions(restarts=1 if args.quick else 5, n_samples=20 if args.quick else 100))
    result = m.run_experiment(records, curve, config)

    print(f"records: {result.preparation}")
    print("\nDeterministic metrics, mean over turbines (Eqs. 13-17; NRMSE and NMAE as fractions of x_max):")
    print(result.summary().to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nProbabilistic metrics, turbine {result.turbines[0]} (Eqs. 18-20):")
    print(result.probabilistic.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
