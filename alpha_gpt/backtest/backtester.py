"""Simple long-short quintile portfolio backtester.

Ranks stocks by alpha signal, goes long top quintile and short bottom quintile,
computes daily PnL and standard performance metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Results from backtesting an alpha signal."""
    portfolio_returns: pd.Series     # Daily long-short returns
    cumulative_returns: pd.Series    # Equity curve (cumulative product)
    sharpe: float                    # Annualized Sharpe ratio
    max_drawdown: float              # Peak-to-trough max drawdown
    annual_return: float             # Annualized return
    quantile_returns: pd.DataFrame   # Mean return per quantile (check monotonicity)


def backtest_alpha(
    alpha_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
) -> BacktestResult:
    """Backtest an alpha signal using long-short quintile portfolio.

    Args:
        alpha_values: Panel of alpha signal values (index=date, columns=PERMNO).
        forward_returns: Panel of next-day returns (same shape).
        n_quantiles: Number of quantile buckets (default 5 = quintiles).

    Returns:
        BacktestResult with portfolio returns and metrics.
    """
    # Align dates
    common_dates = alpha_values.index.intersection(forward_returns.index)
    alpha_values = alpha_values.loc[common_dates]
    forward_returns = forward_returns.loc[common_dates]

    daily_returns = []
    quantile_buckets = {q: [] for q in range(1, n_quantiles + 1)}

    for date in common_dates:
        alpha_row = alpha_values.loc[date].dropna()
        ret_row = forward_returns.loc[date].dropna()
        common_stocks = alpha_row.index.intersection(ret_row.index)

        if len(common_stocks) < n_quantiles * 2:
            daily_returns.append(0.0)
            continue

        alpha_day = alpha_row[common_stocks]
        ret_day = ret_row[common_stocks]

        # Assign quantiles (1 = lowest alpha, n = highest alpha)
        quantiles = pd.qcut(alpha_day, n_quantiles, labels=False, duplicates="drop") + 1

        # Long top quantile, short bottom quantile
        long_mask = quantiles == n_quantiles
        short_mask = quantiles == 1

        long_ret = ret_day[long_mask].mean() if long_mask.sum() > 0 else 0.0
        short_ret = ret_day[short_mask].mean() if short_mask.sum() > 0 else 0.0
        ls_ret = long_ret - short_ret
        daily_returns.append(ls_ret)

        # Track quantile returns
        for q in range(1, n_quantiles + 1):
            q_mask = quantiles == q
            if q_mask.sum() > 0:
                quantile_buckets[q].append(ret_day[q_mask].mean())

    # Build return series
    portfolio_returns = pd.Series(daily_returns, index=common_dates, name="ls_return")
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Metrics
    sharpe = _annualized_sharpe(portfolio_returns)
    max_dd = _max_drawdown(cumulative_returns)
    annual_ret = _annualized_return(portfolio_returns)

    # Quantile returns summary
    quantile_returns = pd.DataFrame({
        f"Q{q}": {"mean_daily_return": np.mean(rets) if rets else 0.0,
                   "count": len(rets)}
        for q, rets in quantile_buckets.items()
    }).T

    return BacktestResult(
        portfolio_returns=portfolio_returns,
        cumulative_returns=cumulative_returns,
        sharpe=sharpe,
        max_drawdown=max_dd,
        annual_return=annual_ret,
        quantile_returns=quantile_returns,
    )


def _annualized_sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio (assuming 252 trading days)."""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def _max_drawdown(cumulative: pd.Series) -> float:
    """Maximum drawdown from peak."""
    if cumulative.empty:
        return 0.0
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def _annualized_return(returns: pd.Series) -> float:
    """Annualized return from daily returns."""
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    n_years = len(returns) / 252
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)
