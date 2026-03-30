"""Alpha-GPT 3.0: End-to-end pipeline orchestrator.

Usage:
    python -m alpha_gpt.main "stocks with high momentum and low valuation tend to outperform"
"""

import sys
import logging

from openai import OpenAI

from alpha_gpt.config import Config
from alpha_gpt.data.loader import load_panels, split_data
from alpha_gpt.debate.moderator import run_debate
from alpha_gpt.gp_search.seed_injector import inject_seeds
from alpha_gpt.gp_search.primitives import create_primitive_set, DEFAULT_TERMINALS
from alpha_gpt.gp_search.engine import run_gp, _eval_expr
from alpha_gpt.backtest.backtester import backtest_alpha
from alpha_gpt.analysis.metrics import compute_ic, compute_icir, compute_turnover
from alpha_gpt.analysis.explainer import explain_alpha

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(trading_idea: str, config: Config | None = None) -> list[dict]:
    """Run the full Alpha-GPT 3.0 pipeline.

    Args:
        trading_idea: Natural language trading idea.
        config: Pipeline configuration (uses defaults if None).

    Returns:
        List of top-K alpha results with expressions, metrics, and explanations.
    """
    config = config or Config()

    # --- Step 1: Load data ---
    print("\n=== Step 1: Loading data ===")
    panels = load_panels(config.data_dir)
    train, val, test = split_data(panels, config.train_end, config.val_end)

    # --- Step 2: Debate → seed alphas ---
    print("\n=== Step 2: Multi-agent debate ===")
    client = OpenAI(
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
    )

    seed_proposals = run_debate(
        trading_idea=trading_idea,
        client=client,
        model=config.llm_model,
        num_rounds=config.debate_rounds,
    )

    # Parse seed expressions into DEAP trees
    terminal_names = [name for name in DEFAULT_TERMINALS if name in train.panels]
    pset = create_primitive_set(terminal_names)
    seed_expressions = [p["expression"] for p in seed_proposals if "expression" in p]
    seed_trees = inject_seeds(seed_expressions, pset)

    print(f"\nParsed {len(seed_trees)}/{len(seed_expressions)} seed expressions into GP trees")

    # --- Step 3: GP search ---
    print("\n=== Step 3: Genetic programming search ===")
    evolved = run_gp(
        seed_trees=seed_trees,
        data=train,
        population_size=config.gp_population,
        generations=config.gp_generations,
        crossover_prob=config.gp_crossover,
        mutation_prob=config.gp_mutation,
        tournament_size=config.gp_tournament_size,
        max_depth=config.gp_max_depth,
    )

    # --- Step 4: Evaluate top-K on test set ---
    print(f"\n=== Step 4: Evaluating top {config.top_k} alphas on test set ===")
    results = []
    for alpha_dict in evolved[:config.top_k]:
        expr_str = alpha_dict["expression"]
        tree = alpha_dict["tree"]

        # Evaluate alpha on test set
        alpha_values = _eval_expr(tree, pset, test.panels)

        if alpha_values.empty or alpha_values.isna().all().all():
            print(f"  Skipping (all NaN): {expr_str[:60]}")
            continue

        # Metrics
        ic_series = compute_ic(alpha_values, test.forward_returns)
        ic_mean = float(ic_series.mean()) if not ic_series.empty else 0.0
        icir = compute_icir(ic_series)
        turnover = compute_turnover(alpha_values)

        # Backtest
        bt = backtest_alpha(alpha_values, test.forward_returns)

        # Explain
        explanation = explain_alpha(
            expression=expr_str,
            backtest=bt,
            ic_mean=ic_mean,
            icir=icir,
            client=client,
            model=config.llm_model,
        )

        result = {
            "expression": expr_str,
            "train_ic": alpha_dict["fitness"],
            "test_ic": ic_mean,
            "icir": icir,
            "sharpe": bt.sharpe,
            "annual_return": bt.annual_return,
            "max_drawdown": bt.max_drawdown,
            "turnover": turnover,
            "explanation": explanation,
            "quantile_returns": bt.quantile_returns,
        }
        results.append(result)

        print(f"\n  Alpha: {expr_str[:70]}")
        print(f"    Train IC: {alpha_dict['fitness']:.4f} | Test IC: {ic_mean:.4f} | "
              f"Sharpe: {bt.sharpe:.2f} | Return: {bt.annual_return:.2%}")

    # --- Step 5: Summary ---
    print(f"\n{'='*60}")
    print(f"=== Results: {len(results)} alphas evaluated ===")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"\n#{i+1}: {r['expression'][:80]}")
        print(f"  Test IC={r['test_ic']:.4f} | ICIR={r['icir']:.4f} | "
              f"Sharpe={r['sharpe']:.2f} | Return={r['annual_return']:.2%} | "
              f"MaxDD={r['max_drawdown']:.2%}")
        print(f"  {r['explanation'][:200]}")

    return results


if __name__ == "__main__":
    idea = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Stocks with high momentum and low valuation tend to outperform"
    print(f"Trading idea: {idea}")
    run_pipeline(idea)
