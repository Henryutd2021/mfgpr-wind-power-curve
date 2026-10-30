"""The experiment design of the paper's case study (Sections 3-4), for any input data.

1. Prepare the records (``data.prepare_records``).
2. High-fidelity training data: ``n_high`` draws from the KDE of all prepared records of the farm
   (wind speed, power), Section 2.1.
3. Low-fidelity training data: ``n_low`` points of the theoretical power curve.
4. Fit the models (default: nonlinear MF-GPR, linear MF-GPR, standard GPR).
5. Deterministic evaluation (Eqs. 13-17) on the records of each selected turbine.
6. Probabilistic evaluation (Eqs. 18-20, interval width and coverage) on one selected turbine, with
   each model's predictive mean and standard deviation (for the nonlinear model: the first two
   moments of its Monte-Carlo mixture) as a Gaussian predictive distribution.

``run_experiment`` returns all tables; ``ExperimentResult.save`` writes them as CSV files.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .data import low_fidelity_points, prepare_records, validate_power_curve
from .kde import kde_sample
from .models import KINDS, LABELS, ModelOptions, make_model


@dataclass
class ExperimentConfig:
    n_high: int = 200                    # KDE draws used as high-fidelity training data
    n_low: int = 200                     # points of the theoretical curve used as low-fidelity data
    kde_bandwidth: float | tuple | None = None  # per (wind speed, power); None: Scott's rule
    kde_lower: tuple | None = None       # e.g. (0, 0) to reject negative draws; None: no limits
    models: tuple = KINDS
    turbines: tuple | None = None        # turbines to evaluate; None: see n_turbines
    n_turbines: int | None = None        # random selection of this many turbines; None: all
    probabilistic_turbine: object = None  # turbine for the probabilistic evaluation; None: first selected
    x_max: float | None = None           # normaliser of Eqs. 14 and 16; None: maximum of the theoretical curve
    n_bins: int = 10                     # categories of Eq. 19: equal bins between 0 and x_max
    mape_min_actual: float = 0.0         # Eq. 17 uses records with |power| above this value (kW)
    quantiles: tuple = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))
    nominal: tuple = tuple(np.round(np.arange(0.1, 0.91, 0.1), 2))
    drop_negative_power: bool = True
    seed: int = 0
    model_options: ModelOptions = field(default_factory=ModelOptions)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    preparation: dict
    turbines: list
    training_high: pd.DataFrame
    training_low: pd.DataFrame
    deterministic: pd.DataFrame
    probabilistic: pd.DataFrame
    intervals: pd.DataFrame
    models: dict

    def summary(self) -> pd.DataFrame:
        """Deterministic metrics averaged over the evaluated turbines, one row per model."""
        cols = ["rmse", "nrmse", "mae", "nmae", "mape_percent"]
        return self.deterministic.groupby("model", sort=False)[cols].mean().reset_index()

    def save(self, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.training_high.to_csv(d / "training_high.csv", index=False)
        self.training_low.to_csv(d / "training_low.csv", index=False)
        self.deterministic.to_csv(d / "deterministic.csv", index=False)
        self.summary().to_csv(d / "deterministic_mean.csv", index=False)
        self.probabilistic.to_csv(d / "probabilistic.csv", index=False)
        self.intervals.to_csv(d / "intervals.csv", index=False)
        cfg = asdict(self.config)
        meta = {"config": cfg, "preparation": self.preparation, "turbines": [str(t) for t in self.turbines]}
        (d / "run.json").write_text(json.dumps(meta, indent=2, default=str))
        return d


def _select_turbines(ids, config, rng):
    ids = list(pd.unique(ids))
    if config.turbines is not None:
        missing = [t for t in config.turbines if t not in ids]
        if missing:
            raise ValueError(f"turbines not in the records: {missing}")
        return list(config.turbines)
    if config.n_turbines is None or config.n_turbines >= len(ids):
        return ids
    pick = rng.choice(len(ids), size=config.n_turbines, replace=False)
    return [ids[i] for i in sorted(pick)]


def run_experiment(records: pd.DataFrame, curve: pd.DataFrame, config: ExperimentConfig | None = None,
                   eval_records: pd.DataFrame | None = None) -> ExperimentResult:
    """Run the design above. ``eval_records`` (optional) are evaluated instead of ``records``."""
    config = config or ExperimentConfig()
    unknown = [k for k in config.models if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown model kind(s) {unknown}; expected some of {KINDS}")
    rng = np.random.default_rng(config.seed)
    prepared, counts = prepare_records(records, config.drop_negative_power)
    evaluated = prepared if eval_records is None else prepare_records(eval_records, config.drop_negative_power)[0]
    curve = validate_power_curve(curve)
    x_max = float(curve["power"].max()) if config.x_max is None else float(config.x_max)

    xy = prepared[["wind_speed", "power"]].to_numpy()
    high = kde_sample(xy, config.n_high, config.kde_bandwidth, rng=rng, lower=config.kde_lower)
    s_low, p_low = low_fidelity_points(curve, config.n_low)
    levels = [(s_low, p_low), (high[:, 0], high[:, 1])]

    models = {}
    for kind in config.models:
        models[kind] = make_model(kind, config.model_options).fit(levels)

    turbines = _select_turbines(evaluated["turbine_id"], config, rng)
    det_rows = []
    for t in turbines:
        rec = evaluated[evaluated["turbine_id"] == t]
        for kind, model in models.items():
            mean, _ = model.predict(rec["wind_speed"].to_numpy())
            row = {"turbine_id": t, "model": LABELS[kind], "n_records": len(rec)}
            row.update(metrics.deterministic_metrics(rec["power"].to_numpy(), mean, x_max, config.mape_min_actual))
            det_rows.append(row)

    prob_turbine = turbines[0] if config.probabilistic_turbine is None else config.probabilistic_turbine
    rec = evaluated[evaluated["turbine_id"] == prob_turbine]
    if rec.empty:
        raise ValueError(f"no records for the probabilistic turbine {prob_turbine!r}")
    edges = np.linspace(0.0, x_max, config.n_bins + 1)
    prob_rows, int_rows = [], []
    for kind, model in models.items():
        mean, std = model.predict(rec["wind_speed"].to_numpy(), include_noise=True)
        y = rec["power"].to_numpy()
        row = {"turbine_id": prob_turbine, "model": LABELS[kind]}
        row.update(metrics.probabilistic_metrics(y, mean, std, edges, config.quantiles))
        prob_rows.append(row)
        for r in metrics.interval_curves(y, mean, std, config.nominal):
            int_rows.append({"turbine_id": prob_turbine, "model": LABELS[kind], **r})

    return ExperimentResult(
        config=config, preparation=counts, turbines=turbines,
        training_high=pd.DataFrame({"wind_speed": high[:, 0], "power": high[:, 1]}),
        training_low=pd.DataFrame({"wind_speed": s_low, "power": p_low}),
        deterministic=pd.DataFrame(det_rows), probabilistic=pd.DataFrame(prob_rows),
        intervals=pd.DataFrame(int_rows), models=models)
