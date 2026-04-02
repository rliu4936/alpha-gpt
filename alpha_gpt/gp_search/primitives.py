"""DEAP PrimitiveSet definition for alpha expression trees.

Uses a single type (panel DataFrame) for all inputs/outputs.
Operators are curried with fixed window sizes so DEAP doesn't need
to evolve integer constants.
"""

import pandas as pd
from deap import gp

from alpha_gpt.operators.alpha_ops import (
    # Unary time-series
    ts_mean_5, ts_mean_10, ts_mean_20, ts_mean_60,
    ts_std_5, ts_std_10, ts_std_20, ts_std_60,
    ts_delta_5, ts_delta_10, ts_delta_20,
    ts_rank_5, ts_rank_10, ts_rank_20,
    ts_min_5, ts_min_10, ts_min_20,
    ts_max_5, ts_max_10, ts_max_20,
    ts_returns_1, ts_returns_5, ts_returns_20,
    # Cross-sectional
    cs_rank, cs_zscore,
    # Element-wise unary
    log_abs, abs_val, sign, neg,
    # Binary
    add, sub, mul, safe_div,
    ts_corr_10, ts_corr_20,
)


def create_primitive_set(terminal_names: list[str]) -> gp.PrimitiveSet:
    """Create a DEAP PrimitiveSet with all alpha operators and given terminals.

    Args:
        terminal_names: Names of data fields available as terminals
                       (e.g., ["close", "open", "high", "low", "volume", ...])

    Returns:
        Configured PrimitiveSet ready for use with DEAP GP.
    """
    # Use number of terminals as input arity (they'll be replaced by named terminals)
    pset = gp.PrimitiveSet("alpha", arity=0)

    # --- Register unary operators (1 input) ---
    unary_ops = [
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
    for op in unary_ops:
        pset.addPrimitive(op, 1)

    # --- Register binary operators (2 inputs) ---
    binary_ops = [add, sub, mul, safe_div, ts_corr_10, ts_corr_20]
    for op in binary_ops:
        pset.addPrimitive(op, 2)

    # --- Register terminals (data fields) ---
    # Terminals are added as named constants; actual data is injected at evaluation time
    for name in terminal_names:
        pset.addTerminal(name, name)

    return pset


# Default terminal names (CRSP + key ratios)
DEFAULT_TERMINALS = [
    "close", "open", "high", "low", "volume", "returns",
    "shrout", "market_cap",
    # Compustat ratios
    "bm", "roe", "roa", "pe_op_dil", "ptb", "npm", "gpm",
    "debt_at", "curr_ratio", "accrual",
]
