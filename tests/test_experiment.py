import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import mfgpr_framework as m
from mfgpr_framework.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _config(**kw):
    base = dict(n_high=60, n_low=40, model_options=m.ModelOptions(restarts=1, n_samples=20, seed=0))
    base.update(kw)
    return m.ExperimentConfig(**base)


def test_run_experiment_tables(farm, curve, tmp_path):
    res = m.run_experiment(farm, curve, _config())
    assert len(res.training_high) == 60 and len(res.training_low) == 40
    assert len(res.deterministic) == 3 * 3  # 3 turbines x 3 models
    assert set(res.deterministic["model"]) == set(m.LABELS.values())
    assert len(res.probabilistic) == 3 and len(res.intervals) == 3 * 9
    assert (res.deterministic["nrmse"] == res.deterministic["rmse"] / 2000.0).all()
    assert res.preparation["rows_in"] == len(farm)
    out = res.save(tmp_path / "out")
    for name in ("training_high.csv", "training_low.csv", "deterministic.csv", "deterministic_mean.csv",
                 "probabilistic.csv", "intervals.csv", "run.json"):
        assert (out / name).exists()


def test_selection_and_options(farm, curve):
    res = m.run_experiment(farm, curve, _config(models=("standard_gpr",), n_turbines=2, x_max=1000.0,
                                                kde_lower=(0.0, 0.0), probabilistic_turbine=None))
    assert len(res.turbines) == 2 and set(res.deterministic["model"]) == {"Standard GPR"}
    assert (res.training_high.to_numpy() >= 0).all()
    res2 = m.run_experiment(farm, curve, _config(models=("standard_gpr",), turbines=(2,)),
                            eval_records=farm[farm["turbine_id"] == 2])
    assert res2.turbines == [2]
    with pytest.raises(ValueError):
        m.run_experiment(farm, curve, _config(models=("bogus",)))
    with pytest.raises(ValueError):
        m.run_experiment(farm, curve, _config(turbines=(99,)))


def test_command_line(tmp_path, capsys):
    assert main(["synthetic", str(tmp_path / "in"), "--turbines", "2", "--days", "1"]) == 0
    assert main(["run", str(tmp_path / "in"), "--out-dir", str(tmp_path / "out"), "--n-high", "40",
                 "--n-low", "30", "--restarts", "0", "--n-samples", "10"]) == 0
    printed = capsys.readouterr().out
    assert "Nonlinear MF-GPR" in printed
    assert len(pd.read_csv(tmp_path / "out" / "deterministic.csv")) == 2 * 3


def test_command_line_mape_threshold(tmp_path, capsys):
    # default: Eq. 17 leaves out records at or below 5 % of the rated power, as examples/minimal.py does
    assert main(["synthetic", str(tmp_path / "in"), "--turbines", "2", "--days", "1"]) == 0
    base = ["run", str(tmp_path / "in"), "--n-high", "40", "--n-low", "30", "--restarts", "0",
            "--n-samples", "10", "--models", "standard_gpr"]
    rated = float(m.synthetic_power_curve()["power"].max())

    def run(name, *extra):
        assert main(base + ["--out-dir", str(tmp_path / name), *extra]) == 0
        cfg = json.loads((tmp_path / name / "run.json").read_text())["config"]
        mape = pd.read_csv(tmp_path / name / "deterministic.csv")["mape_percent"]
        return cfg["mape_min_actual"], mape, capsys.readouterr().out

    threshold, mape_default, printed = run("default")
    assert threshold == pytest.approx(0.05 * rated)
    assert "5 % of the rated power" in printed
    assert run("fraction", "--mape-min-fraction", "0.05")[0] == pytest.approx(threshold)
    assert run("fraction10", "--mape-min-fraction", "0.1")[0] == pytest.approx(0.1 * rated)
    zero, mape_zero, _ = run("absolute", "--mape-min-actual", "0")
    assert zero == 0.0
    assert run("absolute50", "--mape-min-actual", "50")[0] == 50.0
    assert (mape_default < mape_zero).all()  # records near zero power inflate the relative error
    for bad in (["--mape-min-fraction", "0.1", "--mape-min-actual", "5"], ["--mape-min-fraction", "1"],
                ["--mape-min-actual", "-1"]):
        with pytest.raises(SystemExit):
            main(base + ["--out-dir", str(tmp_path / "bad"), *bad])


def test_minimal_example_runs():
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "minimal.py"), "--quick"],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "Standard GPR" in proc.stdout
