# Alpha-GPT 3.0 MVP Plan

## Context

We're building Alpha-GPT 3.0: a multi-agent LLM framework that discovers quantitative trading alphas (symbolic formulas predicting stock returns). The novel contribution is a **debate framework** where multiple LLM agents argue over trading ideas before generating seed formulas, which are then evolved via genetic programming. We have CRSP daily prices (~28GB), Compustat fundamentals (~2.5GB), and pre-computed financial ratios (~570MB) as data.

**Goal:** End-to-end MVP that takes a trading idea in natural language and produces evaluated alpha formulas with quant metrics.

---

## Project Structure

```
alpha_gpt/
├── config.py                  # Dataclass config (LLM keys, GP params, data paths)
├── main.py                    # run_pipeline(trading_idea) orchestrator
│
├── data/
│   ├── loader.py              # Load parquet panels, split train/val/test
│   └── prepare.py             # One-time: CSV → subsetted parquet
│
├── operators/
│   └── alpha_ops.py           # All operators (ts, cs, element-wise) on panel DataFrames
│
├── debate/
│   ├── agents.py              # Debate agent personas + LLM calls
│   ├── moderator.py           # Orchestrate 2-round debate, deduplicate & validate
│   └── prompts.py             # System/user prompt templates
│
├── gp_search/
│   ├── primitives.py          # DEAP PrimitiveSet (curried operators + terminals)
│   ├── engine.py              # GP loop (DEAP eaSimple, fitness = IC)
│   └── seed_injector.py       # Parse LLM expression strings → DEAP trees
│
├── backtest/
│   └── backtester.py          # Long-short quintile portfolio simulator
│
├── analysis/
│   ├── metrics.py             # IC, ICIR, turnover
│   └── explainer.py           # LLM explains evolved alphas in plain language
│
└── utils.py                   # Shared helpers
```

```
scripts/
└── prepare_data.py            # Entry point for data preparation
```

---

## Phase 1: Data Pipeline

**Files:** `scripts/prepare_data.py`, `alpha_gpt/data/loader.py`

### prepare_data.py (run once)
- Read CRSP CSV in chunks, keep only: PERMNO, YYYYMMDD, DlyRet, DlyClose, DlyOpen, DlyHigh, DlyLow, DlyVol, ShrOut
- Filter to 2010-2023, drop penny stocks (price < $5), require minimum trading days
- Subsample ~500 stocks stratified by market cap for fast GP evaluation
- Pivot to panel format: index=date, columns=PERMNO, one parquet per field
- Read Compustat ratios CSV, keep key ratios (pe, pb, roe, bm, npm, etc.), merge on PERMNO+date
- Save all panels to `data/panels/`

### loader.py
- `load_panels(data_dir) → dict[str, pd.DataFrame]` — load all parquet panels
- `split_data(panels, train_end, val_end)` — split into train (2010-2017), val (2018-2020), test (2021-2023)
- Precompute `forward_returns` panel (next-day returns) for IC calculation

**Verify:** Load panels, check shapes align, no look-ahead bias in forward returns.

---

## Phase 2: Operators

**File:** `alpha_gpt/operators/alpha_ops.py`

All operators take and return `pd.DataFrame` (T×N panel). Curried with fixed windows for DEAP compatibility.

### Time-series (per-stock, along time axis)
- `ts_mean_{5,10,20,60}(x)` — rolling mean
- `ts_std_{5,10,20,60}(x)` — rolling std
- `ts_delta_{5,10,20}(x)` — x - x.shift(d)
- `ts_rank_{5,10,20}(x)` — rolling rank
- `ts_corr_{10,20}(x, y)` — rolling correlation
- `ts_min_{5,10,20}(x)`, `ts_max_{5,10,20}(x)`
- `ts_returns_{1,5,20}(x)` — pct_change

### Cross-sectional (across stocks, per date)
- `cs_rank(x)` — rank across stocks each day
- `cs_zscore(x)` — z-score across stocks each day

### Element-wise
- `add(x, y)`, `sub(x, y)`, `mul(x, y)`, `safe_div(x, y)`
- `log(x)`, `abs(x)`, `sign(x)`, `neg(x)`

All ops must handle NaN gracefully (propagate, don't crash).

**Verify:** Unit test each operator on small synthetic panel.

---

## Phase 3: GP Search Infrastructure

**Files:** `alpha_gpt/gp_search/primitives.py`, `engine.py`, `seed_injector.py`

### primitives.py
- Define DEAP `PrimitiveSet` with single type (panel DataFrame)
- Register all curried operators as primitives
- Register terminals: `close`, `open`, `high`, `low`, `volume`, `returns`, `shrout` + ratio fields like `bm`, `roe`, `pe`
- Set depth limits (min=2, max=6) to keep expressions interpretable

### engine.py
- **Fitness function:** Mean Spearman IC between alpha values and forward returns on training set
  - `fitness(individual) → (mean_ic,)` as DEAP maximization
  - Catch evaluation errors → return fitness of 0
- **GP config:** population=100, generations=20, crossover=0.7, mutation=0.2, tournament size=3
- Uses `deap.algorithms.eaSimple` or custom loop with elitism
- Returns top-K individuals sorted by fitness

### seed_injector.py
- Parse LLM expression strings like `cs_rank(ts_delta(close, 5))` into DEAP `PrimitiveTree`
- Map generic operator names to curried versions (e.g., `ts_delta(x, 5)` → `ts_delta_5(x)`)
- Fallback: if parsing fails, log warning and use random individual instead
- Inject seeds into initial population (replace worst individuals)

**Verify:** Create a known-good alpha (e.g., `cs_rank(ts_returns_20(close))`), evaluate fitness, confirm positive IC. Run GP for 5 generations, confirm fitness improves.

---

## Phase 4: Debate Framework (Novel Contribution)

**Files:** `alpha_gpt/debate/agents.py`, `moderator.py`, `prompts.py`

### agents.py
Three debate agents with distinct personas:
1. **Momentum Agent** — favors trend-following signals (price momentum, volume breakouts)
2. **Mean-Reversion Agent** — favors contrarian signals (oversold bounces, ratio extremes)
3. **Fundamental Agent** — favors value/quality signals (earnings, book-to-market, ROE)

Each agent: `generate_alphas(trading_idea, round_num, prior_debate_context) → list[dict]`
- Calls OpenAI/Anthropic API with persona system prompt
- Prompt includes: operator catalog, available data fields, examples of valid expressions
- Returns list of `{expression, description, rationale}`

### moderator.py
- `run_debate(trading_idea, num_rounds=2) → list[dict]`
- **Round 1:** All 3 agents independently propose 2-3 seed alphas each
- **Round 2:** Each agent sees all proposals, critiques others, and can revise or propose new ones
- **Moderation:** LLM call to deduplicate, validate syntax, pick top 5-8 seeds
- Output: validated seed alpha expressions ready for GP injection

### prompts.py
- System prompts for each persona
- Operator catalog reference (copy of available operators + terminals)
- Few-shot examples of valid alpha expressions
- Debate round templates

**Verify:** Run debate on "momentum reversal after earnings" idea, inspect outputs for valid expressions.

---

## Phase 5: Backtesting & Analysis

**Files:** `alpha_gpt/backtest/backtester.py`, `alpha_gpt/analysis/metrics.py`, `explainer.py`

### backtester.py — Simple long-short portfolio simulator
- `backtest_alpha(alpha_values: pd.DataFrame, forward_returns: pd.DataFrame, n_quantiles=5) → BacktestResult`
- Each day:
  1. Rank stocks by alpha signal
  2. Go long top quintile, short bottom quintile (equal-weight within each)
  3. Portfolio return = mean(long returns) - mean(short returns)
- Returns `BacktestResult` dataclass:
  - `portfolio_returns: pd.Series` — daily long-short returns
  - `cumulative_returns: pd.Series` — equity curve
  - `sharpe: float` — annualized Sharpe ratio (√252 × mean/std)
  - `max_drawdown: float` — peak-to-trough max drawdown
  - `annual_return: float` — annualized return
  - `quantile_returns: pd.DataFrame` — mean return per quantile (to check monotonicity)
- No transaction costs for MVP (can add later as a spread parameter)

### metrics.py — Alpha quality metrics (pre-backtest)
- `compute_ic(alpha_values, forward_returns) → pd.Series` — daily Spearman rank IC
- `compute_icir(ic_series) → float` — IC mean / IC std (information ratio)
- `compute_turnover(alpha_values) → float` — average daily rank turnover

### explainer.py
- `explain_alpha(expression, backtest_result, ic_stats) → str`
- LLM call that takes the symbolic expression + backtest metrics + IC stats
- Returns plain-language explanation of what the alpha captures and how it performed

**Verify:** Backtest a known alpha (e.g., 1-month momentum). Expect positive Sharpe, monotonic quantile returns (Q5 > Q1). Compare against naive buy-and-hold.

---

## Phase 6: Orchestration & Integration

**Files:** `alpha_gpt/main.py`, `config.py`

### config.py
```python
@dataclass
class Config:
    data_dir: str = "data/panels"
    # LLM via OpenRouter (OpenAI-compatible API)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-chat-v3-0324"  # or "meta-llama/llama-3.1-70b-instruct"
    # GP params
    gp_population: int = 100
    gp_generations: int = 20
    debate_rounds: int = 2
    top_k: int = 5
    train_end: str = "2017-12-31"
    val_end: str = "2020-12-31"
```

LLM calls use the `openai` SDK with `base_url` pointed to OpenRouter:
```python
client = OpenAI(base_url=config.openrouter_base_url, api_key=os.getenv("OPENROUTER_API_KEY"))
```
`.env` needs: `OPENROUTER_API_KEY=sk-or-...`

### main.py
```python
def run_pipeline(trading_idea: str, config: Config) -> dict:
    # 1. Load data
    panels = load_panels(config.data_dir)
    train, val, test = split_data(panels, ...)

    # 2. Debate → seed alphas
    seeds = run_debate(trading_idea, config)

    # 3. GP search (evolve seeds on train set)
    evolved = run_gp(seeds, train, config)

    # 4. Evaluate top-K on test set
    for alpha in evolved[:config.top_k]:
        alpha.ic_series = compute_ic(alpha.values, test.forward_returns)
        alpha.backtest = backtest_alpha(alpha.values, test.forward_returns)

    # 5. Explain results
    for alpha in evolved[:config.top_k]:
        alpha.explanation = explain_alpha(alpha.expression, alpha.backtest, alpha.ic_series)

    return evolved[:config.top_k]
```

**Verify:** Run full pipeline end-to-end on a simple trading idea. Check that we get alphas with non-zero IC on test set.

---

## Implementation Order

1. **Data pipeline** — everything depends on this
2. **Operators** — needed by GP and by debate prompt examples
3. **GP search** — core search engine, test with random seeds first
4. **Backtester** — long-short portfolio sim, needed to evaluate alphas
5. **Analysis metrics** — IC, ICIR, turnover
6. **Debate framework** — plug in LLM-generated seeds
7. **Explainer** — LLM explains results
8. **Integration** — wire everything together in main.py

---

## Key Libraries (no rebuilding)

| Component | Library | Why |
|-----------|---------|-----|
| GP search | `deap` | Expression trees, crossover, mutation, selection built-in |
| Data | `pandas` + `pyarrow` | Panel operations, parquet I/O |
| IC computation | `scipy.stats.spearmanr` | Rank correlation |
| LLM calls | `openai` SDK via OpenRouter | Single API for any model (DeepSeek V3, Llama 3.1, GPT-4o, etc.) |
| Visualization | `matplotlib` + `seaborn` | IC decay plots, equity curves |

---

## Verification (End-to-End)

1. `python scripts/prepare_data.py` — creates `data/panels/*.parquet`
2. `python -c "from alpha_gpt.data.loader import load_panels; print(load_panels('data/panels').keys())"` — panels load correctly
3. `python -m pytest tests/test_operators.py` — all operators pass
4. `python -c "from alpha_gpt.gp_search.engine import run_gp; ..."` — GP runs and fitness improves
5. `python -m alpha_gpt.main "stocks with high momentum and low valuation tend to outperform"` — full pipeline produces results with IC > 0 on test set
