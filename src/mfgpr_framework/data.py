"""Input data: turbine records (high fidelity) and a theoretical power curve (low fidelity).

Two CSV files form an input directory:

``records.csv`` (one row per turbine and interval)
    turbine_id   identifier of the turbine (integer or text)
    time         optional; interval label (text or timestamp), kept for plotting
    wind_speed   m/s, nacelle wind speed
    power        kW, active power

``power_curve.csv`` (theoretical curve of the turbine type)
    wind_speed   m/s, strictly increasing
    power        kW, >= 0

See docs/data_pipeline.md for how to build them from a SCADA data set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RECORD_COLUMNS = ("turbine_id", "wind_speed", "power")
CURVE_COLUMNS = ("wind_speed", "power")
RECORDS_FILE = "records.csv"
CURVE_FILE = "power_curve.csv"


class DataError(ValueError):
    """Raised when an input table does not follow the schema."""


def _require(df: pd.DataFrame, columns, what: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataError(f"{what}: missing column(s) {missing}")


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema of a records table; values may still be missing (see ``prepare_records``)."""
    _require(df, RECORD_COLUMNS, "records")
    out = df.copy()
    for col in ("wind_speed", "power"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["turbine_id"].isna().any():
        raise DataError("records: turbine_id must not be empty")
    if len(out) == 0:
        raise DataError("records: no rows")
    return out


def prepare_records(df: pd.DataFrame, drop_negative_power: bool = True):
    """Apply the preparation rules and return (prepared table, counts of removed rows).

    Removed: rows whose wind speed or power is missing or not finite, and (by default) rows with
    negative power. Nothing else is filtered: noisy records, curtailment and stops stay in the data,
    because the KDE sampling step is meant to represent the full scatter.
    """
    out = validate_records(df)
    counts = {"rows_in": len(out)}
    finite = np.isfinite(out["wind_speed"]) & np.isfinite(out["power"])
    counts["missing_or_non_finite"] = int((~finite).sum())
    out = out[finite]
    if drop_negative_power:
        neg = out["power"] < 0
        counts["negative_power"] = int(neg.sum())
        out = out[~neg]
    counts["rows_out"] = len(out)
    if len(out) == 0:
        raise DataError("records: no rows left after preparation")
    return out.reset_index(drop=True), counts


def validate_power_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Check a theoretical power-curve table."""
    _require(df, CURVE_COLUMNS, "power curve")
    out = df.loc[:, list(CURVE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if len(out) < 2:
        raise DataError("power curve: need at least two points")
    if not np.all(np.isfinite(out.to_numpy())):
        raise DataError("power curve: values must be finite")
    if np.any(np.diff(out["wind_speed"].to_numpy()) <= 0):
        raise DataError("power curve: wind_speed must be strictly increasing")
    if (out["power"] < 0).any():
        raise DataError("power curve: power must be >= 0")
    return out.reset_index(drop=True)


def low_fidelity_points(curve: pd.DataFrame, n: int, speed_min: float | None = None,
                        speed_max: float | None = None):
    """``n`` evenly spaced points of the theoretical curve (linear interpolation): (speeds, powers)."""
    curve = validate_power_curve(curve)
    if n < 2:
        raise ValueError("n must be at least 2")
    s = curve["wind_speed"].to_numpy()
    lo = s[0] if speed_min is None else speed_min
    hi = s[-1] if speed_max is None else speed_max
    if not lo < hi:
        raise ValueError("speed_min must be below speed_max")
    speeds = np.linspace(lo, hi, n)
    return speeds, np.interp(speeds, s, curve["power"].to_numpy())


def from_sdwpf(df: pd.DataFrame) -> pd.DataFrame:
    """Map a table in the SDWPF layout (TurbID, Day, Tmstamp, Wspd, Patv) to the records schema."""
    _require(df, ("TurbID", "Day", "Tmstamp", "Wspd", "Patv"), "SDWPF table")
    time = pd.to_timedelta(pd.to_numeric(df["Day"]) - 1, unit="D") + pd.to_timedelta(df["Tmstamp"].astype(str) + ":00")
    return pd.DataFrame({"turbine_id": df["TurbID"].to_numpy(), "time": time.astype(str).to_numpy(),
                         "wind_speed": pd.to_numeric(df["Wspd"], errors="coerce").to_numpy(),
                         "power": pd.to_numeric(df["Patv"], errors="coerce").to_numpy()})


def read_inputs(directory):
    """Read and check ``records.csv`` and ``power_curve.csv`` from ``directory``."""
    d = Path(directory)
    for name in (RECORDS_FILE, CURVE_FILE):
        if not (d / name).exists():
            raise DataError(f"{d / name} not found")
    records = validate_records(pd.read_csv(d / RECORDS_FILE))
    curve = validate_power_curve(pd.read_csv(d / CURVE_FILE))
    return records, curve


def write_inputs(directory, records: pd.DataFrame, curve: pd.DataFrame):
    """Write an input directory (checked first)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    validate_records(records).to_csv(d / RECORDS_FILE, index=False)
    validate_power_curve(curve).to_csv(d / CURVE_FILE, index=False)
    return d
