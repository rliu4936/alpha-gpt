"""One-time script: convert raw CSVs to subsetted parquet panel files.

Reads CRSP daily and Compustat ratios CSVs, filters/subsets, pivots to
panel format (index=date, columns=PERMNO), and saves as parquet files
in data/panels/.

Usage:
    python scripts/prepare_data.py
"""

import os
import sys

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(RAW_DIR, "panels")

# Columns to keep from CRSP
CRSP_COLS = [
    "PERMNO", "YYYYMMDD", "DlyRet", "DlyClose", "DlyOpen",
    "DlyHigh", "DlyLow", "DlyVol", "ShrOut", "DlyPrc", "DlyCap",
]

# Date range
DATE_MIN = 20100101
DATE_MAX = 20231231

# Minimum trading days a stock must have to be included
MIN_TRADING_DAYS = 500

# Target number of stocks (stratified sample by market cap)
TARGET_N_STOCKS = 500

# Key ratios to keep from Compustat
RATIO_COLS = [
    "permno", "public_date", "bm", "pe_op_dil", "pe_exi", "ps", "pcf",
    "npm", "opmbd", "gpm", "roa", "roe", "roce", "debt_at", "curr_ratio",
    "quick_ratio", "cash_ratio", "de_ratio", "ptb", "accrual", "divyield",
    "at_turn", "inv_turn", "rect_turn", "GProf", "CAPEI",
]


def load_crsp():
    """Load CRSP daily data in chunks, filter, and return a DataFrame."""
    print("Loading CRSP daily data (this may take a while)...")
    chunks = []
    for chunk in pd.read_csv(
        os.path.join(RAW_DIR, "crsp_daily.csv"),
        usecols=CRSP_COLS,
        chunksize=5_000_000,
        low_memory=False,
    ):
        chunk = chunk[
            (chunk["YYYYMMDD"] >= DATE_MIN) & (chunk["YYYYMMDD"] <= DATE_MAX)
        ]
        chunks.append(chunk)
        print(f"  processed chunk, kept {len(chunk):,} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"CRSP loaded: {len(df):,} rows")

    # Parse date
    df["date"] = pd.to_datetime(df["YYYYMMDD"], format="%Y%m%d")
    df = df.drop(columns=["YYYYMMDD"])

    # Convert all relevant measure columns to numeric to avoid aggregation errors
    num_cols = ["DlyPrc", "DlyRet", "DlyCap", "DlyVol", "DlyClose", "DlyOpen", "DlyHigh", "DlyLow", "ShrOut"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use absolute price (CRSP convention: negative = bid/ask avg)
    df["DlyPrc"] = df["DlyPrc"].abs()

    # Drop penny stocks (price < $5 on average)
    avg_price = df.groupby("PERMNO")["DlyPrc"].mean()
    valid_permnos = avg_price[avg_price >= 5].index
    df = df[df["PERMNO"].isin(valid_permnos)]
    print(f"After dropping penny stocks: {df['PERMNO'].nunique()} stocks")

    # Require minimum trading days
    counts = df.groupby("PERMNO")["date"].count()
    valid_permnos = counts[counts >= MIN_TRADING_DAYS].index
    df = df[df["PERMNO"].isin(valid_permnos)]
    print(f"After min trading days filter: {df['PERMNO'].nunique()} stocks")

    return df


def subsample_stocks(df, n=TARGET_N_STOCKS):
    """Stratified subsample by median market cap."""
    print(f"Subsampling to {n} stocks stratified by market cap...")
    median_cap = df.groupby("PERMNO")["DlyCap"].median().dropna()

    # Split into quintiles and sample proportionally
    quintiles = pd.qcut(median_cap, 5, labels=False, duplicates="drop")
    sampled = []
    per_q = n // 5
    for q in sorted(quintiles.unique()):
        permnos_in_q = quintiles[quintiles == q].index
        sample_n = min(per_q, len(permnos_in_q))
        sampled.extend(
            np.random.RandomState(42).choice(permnos_in_q, sample_n, replace=False)
        )

    df = df[df["PERMNO"].isin(sampled)]
    print(f"Subsampled to {df['PERMNO'].nunique()} stocks")
    return df


def pivot_and_save(df, field, out_dir):
    """Pivot a field to panel format and save as parquet."""
    panel = df.pivot_table(index="date", columns="PERMNO", values=field)
    panel = panel.sort_index()
    path = os.path.join(out_dir, f"{field.lower()}.parquet")
    panel.to_parquet(path)
    print(f"  Saved {field}: {panel.shape}")


def prepare_crsp_panels(df, out_dir):
    """Pivot CRSP fields to panels and save."""
    fields = {
        "DlyClose": "close",
        "DlyOpen": "open",
        "DlyHigh": "high",
        "DlyLow": "low",
        "DlyVol": "volume",
        "DlyRet": "returns",
        "DlyPrc": "price",
        "ShrOut": "shrout",
        "DlyCap": "market_cap",
    }

    for raw_col, panel_name in fields.items():
        if raw_col in df.columns:
            panel = df.pivot_table(index="date", columns="PERMNO", values=raw_col)
            panel = panel.sort_index()
            path = os.path.join(out_dir, f"{panel_name}.parquet")
            panel.to_parquet(path)
            print(f"  Saved {panel_name}: {panel.shape}")


def prepare_ratio_panels(out_dir, valid_permnos):
    """Load Compustat ratios, forward-fill to daily, and save as panels."""
    print("Loading Compustat ratios...")
    ratios_path = os.path.join(RAW_DIR, "compustat_ratios.csv")
    if not os.path.exists(ratios_path):
        print("  compustat_ratios.csv not found, skipping ratios.")
        return

    df = pd.read_csv(ratios_path, usecols=RATIO_COLS, low_memory=False)
    df["date"] = pd.to_datetime(df["public_date"])
    df = df.drop(columns=["public_date"])
    df = df.rename(columns={"permno": "PERMNO"})

    # Filter to our stock universe
    df = df[df["PERMNO"].isin(valid_permnos)]
    print(f"  Ratios: {len(df):,} rows, {df['PERMNO'].nunique()} stocks")

    # For each ratio, pivot to panel and forward-fill (ratios are quarterly)
    ratio_fields = [c for c in df.columns if c not in ("PERMNO", "date")]
    for field in ratio_fields:
        df[field] = pd.to_numeric(df[field], errors="coerce")
        panel = df.pivot_table(index="date", columns="PERMNO", values=field)
        panel = panel.sort_index()

        # Reindex to daily and forward-fill (quarterly ratios applied daily)
        close_path = os.path.join(out_dir, "close.parquet")
        if os.path.exists(close_path):
            daily_index = pd.read_parquet(close_path).index
            panel = panel.reindex(daily_index).ffill()

        path = os.path.join(out_dir, f"{field.lower()}.parquet")
        panel.to_parquet(path)
        print(f"  Saved ratio {field}: {panel.shape}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Load and filter CRSP
    df = load_crsp()

    # Step 2: Subsample stocks
    df = subsample_stocks(df)

    # Step 3: Save CRSP panels
    print("Saving CRSP panels...")
    prepare_crsp_panels(df, OUT_DIR)

    # Step 4: Save forward returns (next-day returns for IC calculation)
    print("Computing forward returns...")
    returns_panel = pd.read_parquet(os.path.join(OUT_DIR, "returns.parquet"))
    forward_returns = returns_panel.shift(-1)  # next-day return
    forward_returns.to_parquet(os.path.join(OUT_DIR, "forward_returns.parquet"))
    print(f"  Saved forward_returns: {forward_returns.shape}")

    # Step 5: Compustat ratios
    valid_permnos = df["PERMNO"].unique()
    prepare_ratio_panels(OUT_DIR, valid_permnos)

    print(f"\nDone! Panels saved to {OUT_DIR}/")
    print(f"Files: {os.listdir(OUT_DIR)}")


if __name__ == "__main__":
    main()
