# MF-GPR framework: probabilistic wind power curves from a theoretical curve and measured data

Author: [Honglin Li](https://henryutd2021.github.io/)

A documented, runnable implementation of the method in:

> Honglin Li, Cong Feng and Jie Zhang. **A Multi-Fidelity Gaussian Process Regression Method for Probabilistic Wind Farm Power Curve Estimation.** *Proceedings of the ASME 2023 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference (IDETC/CIE 2023)*, Volume 3A, V03AT03A034. [doi:10.1115/DETC2023-114762](https://doi.org/10.1115/DETC2023-114762)

The method estimates a power curve as a probability distribution, not a single line. It fuses two sources: the smooth theoretical curve of the turbine type (low fidelity) and measured operating records (high fidelity). A kernel density estimate first condenses the noisy records into a few hundred representative points. Multi-fidelity Gaussian processes, linear and nonlinear, then link the two sources.

## Research question

Can multi-fidelity Gaussian process regression, which combines a theoretical power curve with a small sample of measured data, improve probabilistic estimation of wind turbine power curves?

## Method and scope

| Aspect | In the paper | In this framework |
|---|---|---|
| High-fidelity data | measured records of a wind farm, condensed by kernel density sampling (Section 2.1) | any records in the input schema; `kde_sample` with Scott's rule or a given bandwidth |
| Low-fidelity data | points of the theoretical power curve of the turbine type | any tabulated curve; evenly spaced points |
| Models | nonlinear multi-fidelity GPR (Eqs. 10–12), linear multi-fidelity GPR (Eq. 6), standard GPR (Eqs. 4–5) | the same three, behind one interface; any number of fidelity levels |
| Kernels | squared exponential with automatic relevance determination (Eq. 12) | the same, with exact gradients |
| Deterministic metrics | RMSE, NRMSE, MAE, NMAE, MAPE (Eqs. 13–17) | the same |
| Probabilistic metrics | pinball loss, sharpness, reliability (Eqs. 18–20) | the same, plus interval width and coverage over nominal proportions |
| Experiment design | turbines selected at random; deterministic metrics per turbine; probabilistic evaluation of one turbine | `ExperimentConfig`, `run_experiment` |

The complete method, with the paper's equation numbers, is in [docs/method.md](docs/method.md).

## What is in this repository

| Path | Content |
|---|---|
| [`docs/method.md`](docs/method.md) | KDE, GP regression, linear and nonlinear multi-fidelity models, metrics, Eqs. (1)–(20); implementation conventions |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | the two inputs, how to build them from a SCADA data set, preparation rules, data checks, input schemas |
| [`docs/experiments.md`](docs/experiments.md) | the case-study design and how to express it; tables behind each figure type; further experiment families |
| [`docs/architecture.md`](docs/architecture.md) | data flow, modules, model parameters, size and run time, output tables, tests |
| [`src/mfgpr_framework/`](src/mfgpr_framework/) | the implementation (NumPy, SciPy, pandas; no other dependency) |
| [`examples/`](examples/) | `minimal.py` (the experiment design, a few seconds), `probabilistic.py` (intervals and probabilistic metrics), `sensitivity.py` (training size and bandwidth), `benchmark_function.py` (linear versus nonlinear fusion on an analytic test function); all on synthetic data |
| [`tests/`](tests/) | 54 tests: equations against direct formulas, exact gradients, model behaviour, data checks, experiment, command line |

## Quick start

```bash
git clone https://github.com/Henryutd2021/mfgpr-wind-power-curve.git
cd mfgpr-wind-power-curve
python3 -m venv .venv && source .venv/bin/activate    # needs Python 3.10+ (check: python3 --version); if older, use e.g. python3.12
pip install -e ".[test]"
python examples/minimal.py                              # synthetic wind farm
pytest
```

With conda instead: `conda env create -f environment.yml && conda activate mfgpr-framework && pip install -e .`

The command line works on input directories:

```bash
mfgpr-framework synthetic my_inputs/                    # write a synthetic input directory
mfgpr-framework run my_inputs/ --out-dir outputs/run    # experiment design, result tables
```

`run` prints the deterministic metrics averaged over the evaluated turbines. The MAPE (Eq. 17) leaves out records at or below 5 % of the rated power, where the relative error is undefined or very large. `--mape-min-fraction` changes this share, and `--mape-min-actual` sets an absolute threshold in kW instead (see [docs/method.md](docs/method.md#32-deterministic-metrics-eqs-1317)).

In Python:

```python
import mfgpr_framework as m

records, curve = m.read_inputs("my_inputs/")           # your own data, see docs/data_pipeline.md
prepared, counts = m.prepare_records(records)
high = m.kde_sample(prepared[["wind_speed", "power"]].to_numpy(), 200, rng=0)
low = m.low_fidelity_points(curve, 200)

model = m.make_model("nonlinear_mf").fit([low, (high[:, 0], high[:, 1])])
mean_kw, std_kw = model.predict([4.0, 8.0, 12.0])
```

**About the example data:** `mfgpr_framework.synthetic` generates an invented turbine type and wind farm. They show how the methods behave and how to read their output. Their numbers say nothing about any real turbine.

## Using your own data

1. Build `records.csv` and `power_curve.csv` as described in [docs/data_pipeline.md](docs/data_pipeline.md). For a data set in the SDWPF layout, `from_sdwpf` maps the columns.
2. Run the experiment design ([docs/experiments.md](docs/experiments.md)) or call the models directly.
3. The output tables are described in [docs/architecture.md](docs/architecture.md#output-tables).

## What this repository does not contain

This repository presents the method as a framework. It does not contain the code, the data or the results of the paper's case study. For the case-study results, see the paper.

## Citation

If you use this framework, please cite the paper above. Citation metadata for both the paper and this software are in [`CITATION.cff`](CITATION.cff).

## Licence

- Code: [MIT](LICENSE).
- Documentation (`README.md`, `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
