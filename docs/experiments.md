# Experiment design

How the paper's case study (Sections 3–4) is organized, and how each part is expressed with the framework. The repository does not contain the case-study data or its results; for those, see the paper.

## The case study

| Element | In the paper | In the framework |
|---|---|---|
| Models compared | nonlinear multi-fidelity GPR (Eqs. 10–12), linear multi-fidelity GPR (Eq. 6), standard GPR on the high-fidelity data (Eqs. 4–5) | `ExperimentConfig.models` (default: all three, `KINDS`) |
| High-fidelity training data | 200 points drawn by KDE from the records of the whole wind farm (Fig. 1) | `n_high`, `kde_bandwidth`, `kde_lower` |
| Low-fidelity training data | 200 points of the theoretical power curve (Fig. 1) | `n_low` |
| Deterministic evaluation | nine randomly selected turbines; RMSE, NRMSE, MAE and NMAE per turbine (Table 2); estimated against measured power over consecutive intervals (Fig. 2) | `n_turbines` (random, seeded) or `turbines`; table `deterministic.csv`; `model.predict` on a turbine's time series |
| Probabilistic evaluation | one selected turbine: predictive intervals of the three models (Fig. 3); sharpness, reliability and pinball loss (Fig. 4) | `probabilistic_turbine`; tables `probabilistic.csv` and `intervals.csv`; `model.predict` on a wind-speed grid |
| Farm-wide view | NRMSE of the nonlinear model for every turbine, on the farm layout (Fig. 5) | `n_turbines=None` (all turbines) and the turbine coordinates of the data set |

## Running it

```python
import mfgpr_framework as m

records, curve = m.read_inputs("my_inputs/")          # see data_pipeline.md
config = m.ExperimentConfig(
    n_high=200, n_low=200,                            # training data, Fig. 1
    n_turbines=9, seed=1,                             # random turbines for the deterministic evaluation
    mape_min_actual=0.05 * curve["power"].max(),      # Eq. 17 without near-zero power (method.md)
    model_options=m.ModelOptions(restarts=10, n_samples=100),
)
result = m.run_experiment(records, curve, config)
result.deterministic        # one row per turbine and model (Eqs. 13-17)
result.probabilistic        # one row per model (Eqs. 18-20) for result.turbines[0]
result.intervals            # width and coverage per nominal proportion
result.save("outputs/case")
```

Command line: `mfgpr-framework run my_inputs/ --out-dir outputs/case --n-turbines 9 --seed 1`.

To evaluate on other records than those used for the KDE draws (for example a later period), pass them as `run_experiment(records, curve, config, eval_records=later_records)`.

## Figures and the tables behind them

| Figure type | Data |
|---|---|
| Training data (high- and low-fidelity points over the curve) | `training_high.csv`, `training_low.csv` |
| Estimated vs measured power over consecutive intervals, per turbine | `model.predict(turbine["wind_speed"])` next to `turbine["power"]`, ordered by `time` |
| Probabilistic power curve with intervals | `model.predict(grid)` → mean ± z·std (see `examples/probabilistic.py`) |
| Sharpness and reliability against nominal proportion | `intervals.csv` (`mean_width`, `coverage`) |
| Pinball loss | `probabilistic.csv` (`pinball`) or `metrics.pinball_loss` per quantile |
| Map of a metric over the farm | `deterministic.csv` joined with the turbine coordinates |

## Further experiment families

The same configuration object covers the usual variations of such a study:

| Family | What varies | How |
|---|---|---|
| Training size | number of KDE draws | `n_high` |
| Smoothing of the KDE | bandwidth per coordinate | `kde_bandwidth=(h_speed, h_power)` |
| Low-fidelity resolution | number of curve points | `n_low` |
| Low-fidelity source | another theoretical or reference curve | another `power_curve.csv` |
| Turbine sample | which turbines are evaluated | `turbines=(…)`, `n_turbines`, `seed` |
| Evaluation period | records used for evaluation | `eval_records` |
| Robustness of the fit | restarts, Monte Carlo samples, random seed | `ModelOptions(restarts=…, n_samples=…, seed=…)` |

A loop over configurations, with `result.summary()` collected per run, gives a sensitivity table; `examples/sensitivity.py` does this for the training size and the KDE bandwidth on synthetic data.
