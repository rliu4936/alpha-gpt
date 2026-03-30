"""Alpha quality metrics: IC, ICIR, turnover."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_ic(alpha_values: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """Compute daily Spearman rank IC between alpha values and forward returns.

    Returns a Series indexed by date with IC values.
    """
    common_dates = alpha_values.index.intersection(forward_returns.index)
    ics = {}

    for date in common_dates:
        a = alpha_values.loc[date].dropna()
        f = forward_returns.loc[date].dropna()
        common = a.index.intersection(f.index)
        if len(common) < 20:
            continue
        ic, _ = spearmanr(a[common], f[common])
        if not np.isnan(ic):
            ics[date] = ic

    return pd.Series(ics, name="IC")


def compute_icir(ic_series: pd.Series) -> float:
    """Information ratio: IC mean / IC std."""
    if ic_series.empty or ic_series.std() == 0:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def compute_turnover(alpha_values: pd.DataFrame) -> float:
    """Average daily rank turnover of the alpha signal.

    Measures how much the cross-sectional ranking changes day to day.
    """
    ranks = alpha_values.rank(axis=1, pct=True)
    daily_changes = ranks.diff().abs().mean(axis=1)
    return float(daily_changes.mean())
