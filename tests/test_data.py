import numpy as np
import pandas as pd
import pytest

import mfgpr_framework as m
from mfgpr_framework.data import DataError


def test_prepare_records_rules():
    df = pd.DataFrame({"turbine_id": [1, 1, 1, 2, 2], "wind_speed": [5.0, np.nan, 7.0, 8.0, 9.0],
                       "power": [300.0, 100.0, -4.0, "x", 900.0]})
    out, counts = m.prepare_records(df)
    assert counts == {"rows_in": 5, "missing_or_non_finite": 2, "negative_power": 1, "rows_out": 2}
    assert list(out["power"]) == [300.0, 900.0]
    kept, _ = m.prepare_records(df, drop_negative_power=False)
    assert len(kept) == 3


def test_schema_errors():
    with pytest.raises(DataError):
        m.validate_records(pd.DataFrame({"turbine_id": [1], "power": [1.0]}))
    with pytest.raises(DataError):
        m.prepare_records(pd.DataFrame({"turbine_id": [1], "wind_speed": [np.nan], "power": [1.0]}))
    bad = [pd.DataFrame({"wind_speed": [1.0, 1.0], "power": [0.0, 1.0]}),
           pd.DataFrame({"wind_speed": [1.0, 2.0], "power": [0.0, -1.0]}),
           pd.DataFrame({"wind_speed": [1.0, np.inf], "power": [0.0, 1.0]}),
           pd.DataFrame({"wind_speed": [1.0], "power": [0.0]})]
    for df in bad:
        with pytest.raises(DataError):
            m.validate_power_curve(df)


def test_low_fidelity_points(curve):
    s, p = m.low_fidelity_points(curve, 5, 0.0, 12.0)
    np.testing.assert_allclose(s, [0, 3, 6, 9, 12])
    np.testing.assert_allclose(p, np.interp(s, curve["wind_speed"], curve["power"]))
    with pytest.raises(ValueError):
        m.low_fidelity_points(curve, 1)


def test_from_sdwpf_layout():
    raw = pd.DataFrame({"TurbID": [1, 1], "Day": [1, 2], "Tmstamp": ["00:00", "00:10"],
                        "Wspd": [6.1, 7.2], "Wdir": [0, 0], "Patv": [400.0, None]})
    rec = m.from_sdwpf(raw)
    assert list(rec.columns) == ["turbine_id", "time", "wind_speed", "power"]
    assert rec["time"].tolist() == [str(pd.Timedelta(0)), str(pd.Timedelta(days=1, minutes=10))]
    out, counts = m.prepare_records(rec)
    assert counts["missing_or_non_finite"] == 1 and len(out) == 1


def test_input_directory_round_trip(tmp_path, farm, curve):
    m.write_inputs(tmp_path, farm, curve)
    records, c = m.read_inputs(tmp_path)
    assert len(records) == len(farm)
    pd.testing.assert_frame_equal(c, m.validate_power_curve(curve))
    with pytest.raises(DataError):
        m.read_inputs(tmp_path / "missing")


def test_synthetic_data_is_reproducible(curve):
    a = m.synthetic_farm(2, 1, seed=5)
    b = m.synthetic_farm(2, 1, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["turbine_id"]) == {1, 2} and len(a) == 2 * 144
    assert curve["power"].max() == 2000.0
    with pytest.raises(ValueError):
        m.synthetic_farm(0, 1)
