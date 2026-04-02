# Alpha-GPT Final Report

**Trading Idea:** 具有强劲近期价格动量且被低估的股票

## Rank 1: `ts_min_20(add(add(ts_max_5(ts_delta_20(roe)), ts_std_5(cs_zscore(curr_ratio))), add(ts_delta_20(roe), roe)))`

- **Test IC:** 0.0417
- **ICIR:** 0.1954
- **Annual Return:** 24.37%
- **Sharpe:** 0.88
- **Max Drawdown:** -39.41%

### AI Explanation
This alpha expression combines two main signals: (1) short-term momentum in ROE (return on equity) changes (captured by the max of recent 5-day ROE deltas) and (2) cross-sectional stability in current ratios (z-scored and smoothed). The logic is that firms with improving profitability (ROE) and stable liquidity (current ratio) tend to outperform. The performance is decent (24.4% annualized return, Sharpe 0.88) but suffers from high drawdowns (-39.4%). The monotonic quantile returns (Q1 to Q5) confirm the signal's predictive power, though the highest quintile (Q5) slightly underperforms Q4, suggesting diminishing returns at extreme alpha values.

---

## Rank 2: `ts_min_20(add(add(add(ts_max_5(ts_delta_20(roe)), ts_std_5(cs_zscore(curr_ratio))), ts_delta_20(roe)), roe))`

- **Test IC:** 0.0417
- **ICIR:** 0.1954
- **Annual Return:** 24.37%
- **Sharpe:** 0.88
- **Max Drawdown:** -39.41%

### AI Explanation
This alpha expression combines short-term and medium-term signals related to profitability (ROE) and liquidity (current ratio). It captures:  
1) **Short-term momentum in ROE** (5-day max of 20-day ROE changes),  
2) **Recent stability in liquidity** (5-day std of cross-sectional z-scored current ratios), and  
3) **Reinforcement of ROE trends** (additional 20-day ROE changes and raw ROE).  

The signal likely works because it identifies firms with improving fundamentals (ROE) while filtering for stable liquidity conditions. The performance shows moderate efficacy (Sharpe 0.88, 24.4% annualized return) but significant drawdowns (-39.4%), suggesting it’s sensitive to market regimes. The monotonic quantile returns confirm the signal’s directional validity.  

*(Key intuition: Combines quality (ROE) and liquidity signals with short-term trend confirmation.)*

---

## Rank 3: `ts_min_20(add(add(ts_max_5(ts_delta_20(roe)), ts_std_5(curr_ratio)), ts_max_5(add(ts_delta_20(roe), roe))))`

- **Test IC:** 0.0412
- **ICIR:** 0.1939
- **Annual Return:** 23.97%
- **Sharpe:** 0.87
- **Max Drawdown:** -39.41%

### AI Explanation
This alpha combines short-term trends and volatility in **return on equity (ROE)** and **current ratio** to identify stocks with improving fundamentals and stability. The signal captures:  
1) **Momentum in ROE** (via 20-day delta and 5-day max windows),  
2) **Liquidity stability** (5-day std of current ratio), and  
3) **Reinforced ROE strength** (adding raw ROE to its delta).  

The logic is that stocks with accelerating profitability (ROE) and steady liquidity (current ratio) may outperform. The performance is decent (Sharpe 0.87, 24% annual return) but suffers from high drawdowns (-39%), suggesting sensitivity to market stress. The monotonic quantile returns (Q1 to Q5) confirm the signal’s directional validity.

---

## Rank 4: `ts_min_20(add(add(ts_max_5(ts_delta_20(roe)), ts_std_5(curr_ratio)), ts_max_5(add(ts_delta_20(ts_delta_20(roe)), roe))))`

- **Test IC:** 0.0415
- **ICIR:** 0.1943
- **Annual Return:** 21.03%
- **Sharpe:** 0.80
- **Max Drawdown:** -36.01%

### AI Explanation
This alpha expression combines several dynamic measures of profitability (ROE) and liquidity (current ratio) to identify stocks with improving fundamentals. The signal captures:  
1) Short-term improvements in ROE (ts_delta_20),  
2) Stability in current ratio (ts_std_5), and  
3) Acceleration in ROE momentum (nested ts_delta_20 terms).  

It likely works because stocks with *sustained fundamental improvement* tend to be rewarded by the market, while the inclusion of liquidity filters avoids overly risky names. The performance shows moderate but consistent predictive power (Sharpe 0.80, positive monotonic quantile returns), though with significant drawdowns (-36%), suggesting it works best as part of a diversified portfolio.  

*(The ts_min_20 wrapper applies a 20-day lookback minimum to smooth the composite signal.)*

---

## Rank 5: `ts_min_20(add(add(ts_std_5(ts_mean_60(neg(curr_ratio))), roe), ts_max_5(add(ts_delta_20(roe), ts_max_5(ts_delta_20(roe))))))`

- **Test IC:** 0.0403
- **ICIR:** 0.1942
- **Annual Return:** 29.04%
- **Sharpe:** 1.01
- **Max Drawdown:** -38.21%

### AI Explanation
This alpha combines **short-term mean-reversion** (via negative current ratio and its rolling stats) with **momentum in profitability** (via ROE and its derivatives). The signal likely profits from stocks that are temporarily oversold (low current ratio) but have improving fundamentals (rising ROE), while avoiding those with deteriorating ROE. The performance suggests moderate but consistent predictive power (Sharpe 1.01, positive IC), though the strategy suffers during drawdowns (-38.2%). The quantile returns show monotonicity, with Q4 surprisingly outperforming Q5—hinting at potential overfitting or nonlinear effects in the highest-alpha stocks.  

*(Note: The current ratio inversion (`neg(curr_ratio)`) suggests the signal bets on firms with weak liquidity, implying a possible distress-risk premium or short-term bounce effect.)*

---

