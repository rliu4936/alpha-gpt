"""Load parquet panel data and split into train/val/test sets."""

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataSplit:
    """Container for panel data split into time periods."""
    panels: dict[str, pd.DataFrame]
    forward_returns: pd.DataFrame


def load_panels(data_dir: str = "data/panels") -> dict[str, pd.DataFrame]:
    """Load all parquet panel files from data_dir.

    Returns dict mapping field name to DataFrame (index=date, columns=PERMNO).
    """
    panels = {}
    for f in os.listdir(data_dir):
        if f.endswith(".parquet"):
            name = f.replace(".parquet", "")
            panels[name] = pd.read_parquet(os.path.join(data_dir, f))
    print(f"Loaded {len(panels)} panels: {list(panels.keys())}")
    return panels


def split_data(
    panels: dict[str, pd.DataFrame],
    train_end: str = "2017-12-31",
    val_end: str = "2020-12-31",
) -> tuple[DataSplit, DataSplit, DataSplit]:
    """Split panels into train, validation, and test sets by date."""
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    def _slice(p: dict[str, pd.DataFrame], start, end) -> DataSplit:
        sliced = {}
        for name, df in p.items():
            if name == "forward_returns":
                continue
            if start is None:
                mask = (df.index <= end)
            elif end is None:
                mask = (df.index >= start)
            else:
                mask = (df.index >= start) & (df.index <= end)
            sliced[name] = df.loc[mask]

        fwd = p.get("forward_returns", pd.DataFrame())
        if not fwd.empty:
            if start is None:
                mask = (fwd.index <= end)
            elif end is None:
                mask = (fwd.index >= start)
            else:
                mask = (fwd.index >= start) & (fwd.index <= end)
            fwd = fwd.loc[mask]

        return DataSplit(panels=sliced, forward_returns=fwd)

    train = _slice(panels, None, train_end_dt)
    val = _slice(panels, train_end_dt + pd.Timedelta(days=1), val_end_dt)
    test = _slice(panels, val_end_dt + pd.Timedelta(days=1), None)

    # For test, use all remaining dates
    test_panels = {}
    test_fwd = panels.get("forward_returns", pd.DataFrame())
    for name, df in panels.items():
        if name == "forward_returns":
            continue
        test_panels[name] = df.loc[df.index > val_end_dt]
    if not test_fwd.empty:
        test_fwd = test_fwd.loc[test_fwd.index > val_end_dt]
    test = DataSplit(panels=test_panels, forward_returns=test_fwd)

    print(f"Train: {_date_range(train)}, Val: {_date_range(val)}, Test: {_date_range(test)}")
    return train, val, test


def _date_range(split: DataSplit) -> str:
    """Helper to format date range string."""
    sample = next(iter(split.panels.values()), pd.DataFrame())
    if sample.empty:
        return "empty"
    return f"{sample.index.min().date()} to {sample.index.max().date()}"
