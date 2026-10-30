# Architecture

## Data flow

```mermaid
flowchart TD
    IN[input directory<br/>records.csv, power_curve.csv] --> IO[data.read_inputs]
    IO --> PREP[data.prepare_records]
    PREP --> KDE[kde.kde_sample]
    IO --> LF[data.low_fidelity_points]
    KDE --> FIT[models.make_model(kind).fit(levels)]
    LF --> FIT
    FIT --> PRED[model.predict: mean, std in kW]
    PRED --> MET[metrics: Eqs. 13-20, interval curves]
    MET --> OUT[experiment.ExperimentResult.save<br/>CSV tables + run.json]
```

`experiment.run_experiment` runs this chain; each step can also be called on its own.

## Modules (`src/mfgpr_framework/`)

| Module | Content |
|---|---|
| `kernels.py` | `RBF` (Eq. 12) on selected input columns, `Sum`, `Product` (Eq. 11); covariance matrices and their exact derivatives with respect to the log-parameters |
| `gp.py` | `GaussianProcess` (Eqs. 4–5): marginal likelihood with gradient, L-BFGS-B with restarts, prediction with or without noise; noise models `GaussianNoise` and `LevelNoise` (one variance per fidelity level) |
| `multifidelity.py` | `LinearMultiFidelityKernel` and `LinearMultiFidelityGP` (Eq. 6, joint over all levels); `NonlinearMultiFidelityGP` (Eqs. 10–12, Monte Carlo propagation); `default_rbf` (starting values and length-scale limits) |
| `models.py` | the three power-curve estimators behind one interface (`StandardGPR`, `LinearMFGPR`, `NonlinearMFGPR`, `make_model`), output scaling, `ModelOptions` |
| `kde.py` | `kde_density` (Eqs. 1–3, product form), `kde_sample`, `scott_bandwidth` |
| `metrics.py` | Eqs. 13–20; Gaussian quantiles; bin probabilities; interval width and coverage |
| `data.py` | input schemas, checks (`DataError`), preparation rules, low-fidelity points, SDWPF column mapping, input directory read/write |
| `experiment.py` | `ExperimentConfig`, `run_experiment`, `ExperimentResult` (tables, summary, `save`) |
| `synthetic.py` | invented turbine type and wind farm for examples and tests |
| `cli.py` | command line `mfgpr-framework synthetic …` and `mfgpr-framework run …` |

## Models and their parameters

| Model | GPs | Inputs of the GP(s) | Hyperparameters |
|---|---|---|---|
| Standard GPR | 1 | wind speed | RBF variance and length scale, noise variance |
| Linear MF-GPR | 1 joint GP over both levels | wind speed + level index | one RBF per level, ρ, one noise variance per level |
| Nonlinear MF-GPR | 1 per level | level 0: wind speed; level 1: wind speed and the level-0 posterior mean | level 0: RBF + noise; level 1: three RBF factors of Eq. (11) (variance of c^f fixed) + noise |

## Size and run time

- A GP solve costs $O(n^3)$ in the number of training points $n$. The linear model works on $n_{\text{low}} + n_{\text{high}}$ points in one joint matrix; the nonlinear model on $n_{\text{low}}$ and $n_{\text{high}}$ separately.
- Prediction with the nonlinear model costs $S$ high-level GP predictions (Monte Carlo samples, default 100) per call; memory and time grow linearly with the number of prediction points.
- On a laptop (one process): `examples/minimal.py` (6 turbines, 4 days, 200 + 200 training points, 5 restarts) takes a few seconds; the tests take about one second.

## Output tables

Written by `ExperimentResult.save(out_dir)`:

| File | Rows | Columns (unit) |
|---|---|---|
| `training_high.csv` | KDE draws | `wind_speed` (m/s), `power` (kW) |
| `training_low.csv` | curve points | `wind_speed` (m/s), `power` (kW) |
| `deterministic.csv` | turbine × model | `turbine_id`, `model`, `n_records`, `rmse` (kW), `nrmse` (–), `mae` (kW), `nmae` (–), `mape_percent` (%) |
| `deterministic_mean.csv` | model | mean of the deterministic metrics over the evaluated turbines |
| `probabilistic.csv` | model | `turbine_id`, `model`, `pinball` (kW, Eq. 18 averaged over quantiles), `sharpness_eq19` (–), `reliability_eq20` (kW) |
| `intervals.csv` | model × nominal proportion | `turbine_id`, `model`, `nominal` (–), `mean_width` (kW), `coverage` (–) |
| `run.json` | – | the configuration, the preparation counts and the evaluated turbines |

## Tests (`tests/`)

- kernels: Eq. (12) against a direct formula; gradients against finite differences (single, product and sum kernels); positive semi-definiteness;
- GP: Eqs. (4)–(5) against a direct solve; gradient of the marginal likelihood; recovery of a smooth function; noise handling; errors;
- multi-fidelity: covariance structure of Eq. (6) for two and three levels; joint-likelihood gradient including ρ; recovery of a known ρ; the nonlinear model on the analytic test function of Perdikaris et al. (2017) against the linear and single-fidelity models; deterministic Monte Carlo prediction; three levels; scaling invariance;
- KDE: Eqs. (1)–(3) by hand, normalisation, Scott's rule, moments of the draws, limits;
- metrics: Eqs. (13)–(19) by hand; closed form of Eq. (20) against numerical integration; interval coverage for calibrated forecasts;
- data and experiment: schema checks, preparation counts, SDWPF mapping, input round trip, experiment tables, command line, minimal example.
