"""Sensitivity of the three models to the training size and the KDE bandwidth (synthetic data).

    python examples/sensitivity.py [--out outputs/sensitivity.csv]

Each run repeats the experiment design with one setting changed and keeps the mean deterministic
metrics over the turbines. The data are invented; the numbers say nothing about real turbines.
"""
import argparse
from pathlib import Path

import pandas as pd

import mfgpr_framework as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/sensitivity.csv")
    args = ap.parse_args()

    records = m.synthetic_farm(n_turbines=4, days=3, seed=2)
    curve = m.synthetic_power_curve()
    cases = {f"n_high={n}": dict(n_high=n) for n in (50, 100, 200, 400)}
    cases.update({f"bandwidth=({hs}, {hp})": dict(kde_bandwidth=(hs, hp)) for hs, hp in ((0.2, 10.0), (1.0, 50.0))})
    rows = []
    for name, change in cases.items():
        config = m.ExperimentConfig(model_options=m.ModelOptions(restarts=2, n_samples=50), **change)
        summary = m.run_experiment(records, curve, config).summary()
        summary.insert(0, "case", name)
        rows.append(summary)
        print(name, "done")
    table = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.pivot(index="case", columns="model", values="rmse").round(1))


if __name__ == "__main__":
    main()
