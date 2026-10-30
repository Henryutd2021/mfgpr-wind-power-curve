"""Probabilistic power curves of one synthetic turbine: interval width and coverage per nominal
proportion, pinball loss, sharpness (Eq. 19) and reliability (Eq. 20), and the curves themselves.

    python examples/probabilistic.py [--out-dir outputs/probabilistic]

Writes curve.csv (mean and central 90 % interval of each model on a wind-speed grid) and
intervals.csv. The data are invented; the numbers say nothing about real turbines.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import mfgpr_framework as m
from mfgpr_framework import metrics as mt
from scipy.stats import norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/probabilistic")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records, _ = m.prepare_records(m.synthetic_farm(n_turbines=4, days=4, seed=1))
    curve = m.synthetic_power_curve()
    rng = np.random.default_rng(0)
    high = m.kde_sample(records[["wind_speed", "power"]].to_numpy(), 200, rng=rng, lower=(0.0, 0.0))
    low = m.low_fidelity_points(curve, 200)
    levels = [low, (high[:, 0], high[:, 1])]

    turbine = records[records["turbine_id"] == 1]
    y = turbine["power"].to_numpy()
    edges = np.linspace(0, curve["power"].max(), 11)
    grid = np.linspace(0, 27, 271)
    curves, rows = [pd.DataFrame({"wind_speed": grid})], []
    for kind in m.KINDS:
        model = m.make_model(kind, m.ModelOptions(restarts=3)).fit(levels)
        mean, std = model.predict(turbine["wind_speed"].to_numpy())
        print(f"{m.LABELS[kind]:<17}", {k: round(v, 4) for k, v in mt.probabilistic_metrics(y, mean, std, edges).items()})
        rows += [{"model": m.LABELS[kind], **r} for r in mt.interval_curves(y, mean, std)]
        gm, gs = model.predict(grid)
        z = norm.ppf(0.95)
        curves.append(pd.DataFrame({f"{kind}_mean": gm, f"{kind}_lo90": gm - z * gs, f"{kind}_hi90": gm + z * gs}))
    pd.concat(curves, axis=1).to_csv(out / "curve.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "intervals.csv", index=False)
    print(pd.DataFrame(rows).pivot(index="nominal", columns="model", values="coverage").round(3))
    print(f"written: {out / 'curve.csv'}, {out / 'intervals.csv'}")


if __name__ == "__main__":
    main()
