# Alpha-GPT 3.0 (Work in Progress)

Multi-agent LLM debate framework for discovering quantitative trading alphas. See [`report_draft.pdf`](report_draft.pdf) for details.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

## Data

Panel data should be in `data/panels/` as parquet files (one per field, each a date x stock DataFrame). To prepare from raw CRSP/Compustat CSVs:

```bash
python scripts/prepare_data.py
```

## Running

### Basic usage

```bash
# Default mode (debate-only): multi-agent debate, direct formula eval, no GP
python -m alpha_gpt.main "Stocks with high momentum and low valuation tend to outperform"
```

### Pipeline modes

There are 5 modes for ablation studies:

| Mode | What it does | LLM? | GP? |
|------|-------------|------|-----|
| `debate-only` | Multi-agent debate, eval seed formulas directly | Yes (20+ calls) | No |
| `full` | Multi-agent debate + GP evolution | Yes (20+ calls) | Yes |
| `single-agent` | One LLM call, eval formulas directly | Yes (1 call) | No |
| `single-agent-gp` | One LLM call + GP evolution | Yes (1 call) | Yes |
| `random-gp` | Pure random GP, no LLM | No | Yes |

```bash
# Full pipeline (debate + GP)
python -m alpha_gpt.main "your trading idea" --mode full

# Single-agent baseline (one LLM call, no debate, no GP)
python -m alpha_gpt.main "your trading idea" --mode single-agent

# Single-agent + GP
python -m alpha_gpt.main "your trading idea" --mode single-agent-gp

# Random GP baseline (no LLM at all)
python -m alpha_gpt.main "your trading idea" --mode random-gp
```

### Multiple runs

For statistical robustness, run each mode multiple times with fixed GP seeds:

```bash
python -m alpha_gpt.main "your trading idea" --mode full --num-runs 3
python -m alpha_gpt.main "your trading idea" --mode debate-only --num-runs 3
python -m alpha_gpt.main "your trading idea" --mode random-gp --num-runs 3
```

Each run saves to `outputs/<timestamp>_<mode>_run<i>/` containing:
- `results.json` — machine-readable metrics (IC, ICIR, Sharpe, etc.)
- `alpha_report.md` — human-readable report
- `equity_curves.png` — cumulative return plot with VW market benchmark
- `gp_evolution.png` — fitness over generations (GP modes only)
- `debate/` — debate artifacts (debate modes only)

### Comparing modes

After running multiple modes, generate a cross-mode comparison:

```bash
python -m alpha_gpt.main --compare
```

This scans all `outputs/*/results.json` and produces `outputs/comparison/summary_table.md` and `outputs/comparison/comparison.json`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -x
```

## Project structure

```
alpha_gpt/
  main.py              # CLI + 5 mode runners + compare
  config.py            # Config dataclass (.env loading)
  data/
    loader.py          # Load parquet panels, train/val/test split
  debate/
    agents.py          # 3 debate agents (Momentum, MeanReversion, Fundamental)
    moderator.py       # Two-stage debate orchestration
    prompts.py         # All prompt templates
    models.py          # Dataclasses for debate objects
  gp_search/
    engine.py          # DEAP GP evolution loop
    primitives.py      # PrimitiveSet (curried operators + terminals)
    seed_injector.py   # Parse LLM expressions into DEAP trees
  operators/
    alpha_ops.py       # 35+ curried operators (ts, cs, element-wise)
  backtest/
    backtester.py      # Long-short quintile backtester
  analysis/
    metrics.py         # IC, ICIR, turnover
    visualize.py       # Equity curves, GP evolution plots
    explainer.py       # LLM explains alphas in plain language
```
