"""Alpha operators on panel DataFrames (index=date, columns=PERMNO).

All operators take and return pd.DataFrame. NaN-safe (propagate, don't crash).
Time-series operators are curried with fixed window sizes for DEAP compatibility.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Time-series operators (per-stock, along the time axis)
# ---------------------------------------------------------------------------

def _make_ts_mean(window):
    def ts_mean(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).mean()
    ts_mean.__name__ = f"ts_mean_{window}"
    return ts_mean

def _make_ts_std(window):
    def ts_std(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=2).std()
    ts_std.__name__ = f"ts_std_{window}"
    return ts_std

def _make_ts_delta(window):
    def ts_delta(x: pd.DataFrame) -> pd.DataFrame:
        return x - x.shift(window)
    ts_delta.__name__ = f"ts_delta_{window}"
    return ts_delta

def _make_ts_rank(window):
    def ts_rank(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).rank(pct=True)
    ts_rank.__name__ = f"ts_rank_{window}"
    return ts_rank

def _make_ts_min(window):
    def ts_min(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).min()
    ts_min.__name__ = f"ts_min_{window}"
    return ts_min

def _make_ts_max(window):
    def ts_max(x: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=1).max()
    ts_max.__name__ = f"ts_max_{window}"
    return ts_max

def _make_ts_returns(window):
    def ts_returns(x: pd.DataFrame) -> pd.DataFrame:
        return x.pct_change(window)
    ts_returns.__name__ = f"ts_returns_{window}"
    return ts_returns

def _make_ts_corr(window):
    def ts_corr(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        return x.rolling(window, min_periods=3).corr(y)
    ts_corr.__name__ = f"ts_corr_{window}"
    return ts_corr


# Generate curried operators for standard windows
WINDOWS_SHORT = [5, 10, 20]
WINDOWS_ALL = [5, 10, 20, 60]

ts_mean_5, ts_mean_10, ts_mean_20, ts_mean_60 = [_make_ts_mean(w) for w in WINDOWS_ALL]
ts_std_5, ts_std_10, ts_std_20, ts_std_60 = [_make_ts_std(w) for w in WINDOWS_ALL]
ts_delta_5, ts_delta_10, ts_delta_20 = [_make_ts_delta(w) for w in WINDOWS_SHORT]
ts_rank_5, ts_rank_10, ts_rank_20 = [_make_ts_rank(w) for w in WINDOWS_SHORT]
ts_min_5, ts_min_10, ts_min_20 = [_make_ts_min(w) for w in WINDOWS_SHORT]
ts_max_5, ts_max_10, ts_max_20 = [_make_ts_max(w) for w in WINDOWS_SHORT]
ts_returns_1, ts_returns_5, ts_returns_20 = [_make_ts_returns(w) for w in [1, 5, 20]]
ts_corr_10, ts_corr_20 = [_make_ts_corr(w) for w in [10, 20]]


# ---------------------------------------------------------------------------
# Cross-sectional operators (across stocks, per date)
# ---------------------------------------------------------------------------

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Rank across stocks each day (percentile rank)."""
    return x.rank(axis=1, pct=True)

def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    """Z-score across stocks each day."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


# ---------------------------------------------------------------------------
# Element-wise operators
# ---------------------------------------------------------------------------

def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x + y

def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x - y

def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x * y

def safe_div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Element-wise division, replacing div-by-zero with NaN."""
    return x / y.replace(0, np.nan)

def log_abs(x: pd.DataFrame) -> pd.DataFrame:
    """Log of absolute value (NaN for zero)."""
    return np.log(x.abs().replace(0, np.nan))

def abs_val(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def neg(x: pd.DataFrame) -> pd.DataFrame:
    return -x


# ---------------------------------------------------------------------------
# Registry: all operators grouped by type for easy access
# ---------------------------------------------------------------------------

# Unary operators (DataFrame -> DataFrame)
UNARY_OPS = [
    ts_mean_5, ts_mean_10, ts_mean_20, ts_mean_60,
    ts_std_5, ts_std_10, ts_std_20, ts_std_60,
    ts_delta_5, ts_delta_10, ts_delta_20,
    ts_rank_5, ts_rank_10, ts_rank_20,
    ts_min_5, ts_min_10, ts_min_20,
    ts_max_5, ts_max_10, ts_max_20,
    ts_returns_1, ts_returns_5, ts_returns_20,
    cs_rank, cs_zscore,
    log_abs, abs_val, sign, neg,
]

# Binary operators (DataFrame, DataFrame -> DataFrame)
BINARY_OPS = [
    add, sub, mul, safe_div,
    ts_corr_10, ts_corr_20,
]

# All operators with metadata
ALL_OPS = {op.__name__: op for op in UNARY_OPS + BINARY_OPS}
