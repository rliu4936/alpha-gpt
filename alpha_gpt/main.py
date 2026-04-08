"""Alpha-GPT 3.0: End-to-end pipeline orchestrator.

Usage:
    python -m alpha_gpt.main "stocks with high momentum and low valuation tend to outperform"
    python -m alpha_gpt.main --debate-only "stocks with high momentum and low valuation tend to outperform"
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

from openai import OpenAI

from alpha_gpt.analysis.explainer import explain_alpha
from alpha_gpt.analysis.metrics import compute_ic, compute_icir, compute_turnover
from alpha_gpt.analysis.visualize import plot_equity_curves, plot_gp_evolution
from alpha_gpt.backtest.backtester import backtest_alpha
from alpha_gpt.config import Config
from alpha_gpt.data.loader import load_panels, split_data
from alpha_gpt.debate.models import to_jsonable
from alpha_gpt.debate.moderator import run_formula_debate, run_idea_debate
from alpha_gpt.gp_search.engine import _eval_expr, run_gp
from alpha_gpt.gp_search.primitives import DEFAULT_TERMINALS, create_primitive_set
from alpha_gpt.gp_search.seed_injector import inject_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
DEFAULT_IDEA = "Stocks with high momentum and low valuation tend to outperform"


def _save_json(path: str, payload) -> None:
    """Persist a JSON-serializable payload with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def _save_debate_artifacts(debate_dir: str, artifacts: dict[str, object]) -> None:
    """Write all debate artifacts to disk."""
    os.makedirs(debate_dir, exist_ok=True)
    for filename, payload in artifacts.items():
        _save_json(os.path.join(debate_dir, filename), payload)


def _create_client(config: Config) -> OpenAI:
    """Create an OpenAI-compatible client configured for OpenRouter."""
    return OpenAI(
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key,
    )


def run_debate_only(
    trading_idea: str,
    config: Config | None = None,
    available_terminals: list[str] | None = None,
) -> list[dict[str, str]]:
    """Run only Stage 1 + Stage 2 debate and return selected seed formulas."""
    config = config or Config()
    terminals = available_terminals or list(DEFAULT_TERMINALS)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Debate-only run. Saving outputs to: {out_dir} ===")
    print(f"\nUsing {len(terminals)} terminals for debate constraints.")

    client = _create_client(config)

    print("\n=== Stage 1: Idea debate ===")
    hypotheses, idea_artifacts = run_idea_debate(
        trading_idea=trading_idea,
        available_terminals=terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )
    print(f"Generated {len(hypotheses)} hypothesis specs.")

    print("\n=== Stage 2: Formula debate ===")
    seed_pack, formula_artifacts = run_formula_debate(
        hypotheses=hypotheses,
        available_terminals=terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )
    print(f"Selected {len(seed_pack.selected_formulas)} seed formulas.")

    if config.save_debate_artifacts:
        debate_artifacts = {}
        debate_artifacts.update(idea_artifacts)
        debate_artifacts.update(formula_artifacts)
        _save_debate_artifacts(os.path.join(out_dir, "debate"), debate_artifacts)

    print("\n=== Debate summary ===")
    for idx, hypothesis in enumerate(hypotheses, start=1):
        title = hypothesis.title or hypothesis.hypothesis_id
        summary = hypothesis.summary.strip()
        print(f"{idx}. {title}")
        if summary:
            print(f"   Summary: {summary}")

    for idx, formula in enumerate(seed_pack.selected_formulas, start=1):
        status = "parseable" if formula.parseable else "not_parseable"
        print(f"{idx}. [{formula.hypothesis_id}] ({formula.formula_role}, {status}) {formula.expression}")

    result = [
        {
            "formula_id": formula.formula_id,
            "hypothesis_id": formula.hypothesis_id,
            "formula_role": formula.formula_role,
            "expression": formula.expression,
            "parseable": str(formula.parseable),
            "description": formula.plain_language_mapping or formula.rationale,
        }
        for formula in seed_pack.selected_formulas
    ]

    summary_payload = {
        "trading_idea": trading_idea,
        "terminals": terminals,
        "hypotheses": hypotheses,
        "selected_formulas": seed_pack.selected_formulas,
        "selection_rationale": seed_pack.selection_rationale,
    }
    _save_json(os.path.join(out_dir, "debate_summary.json"), summary_payload)
    print(f"\nSaved debate summary to {os.path.join(out_dir, 'debate_summary.json')}")
    return result


def run_pipeline(trading_idea: str, config: Config | None = None) -> list[dict]:
    """Run the full Alpha-GPT 3.0 pipeline."""
    config = config or Config()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", timestamp)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Initializing run. Saving outputs to: {out_dir} ===")

    print("\n=== Step 1: Loading data ===")
    panels = load_panels(config.data_dir)
    train, val, test = split_data(panels, config.train_end, config.val_end)
    del val  # Validation remains reserved for future use.

    available_terminals = [name for name in DEFAULT_TERMINALS if name in train.panels]
    pset = create_primitive_set(available_terminals)

    client = _create_client(config)

    print("\n=== Step 2: Idea debate ===")
    hypotheses, idea_artifacts = run_idea_debate(
        trading_idea=trading_idea,
        available_terminals=available_terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )

    print("\n=== Step 3: Formula debate ===")
    seed_pack, formula_artifacts = run_formula_debate(
        hypotheses=hypotheses,
        available_terminals=available_terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )

    if config.save_debate_artifacts:
        debate_artifacts = {}
        debate_artifacts.update(idea_artifacts)
        debate_artifacts.update(formula_artifacts)
        _save_debate_artifacts(os.path.join(out_dir, "debate"), debate_artifacts)

    seed_expressions = [
        formula.expression
        for formula in seed_pack.selected_formulas
        if formula.expression
    ]
    seed_trees = inject_seeds(seed_expressions, pset)
    print(f"\nParsed {len(seed_trees)}/{len(seed_expressions)} seed expressions into GP trees")

    print("\n=== Step 4: Genetic programming search ===")
    evolved, logbook = run_gp(
        seed_trees=seed_trees,
        data=train,
        population_size=config.gp_population,
        generations=config.gp_generations,
        crossover_prob=config.gp_crossover,
        mutation_prob=config.gp_mutation,
        tournament_size=config.gp_tournament_size,
        max_depth=config.gp_max_depth,
    )
    plot_gp_evolution(logbook, os.path.join(out_dir, "gp_evolution.png"))

    print(f"\n=== Step 5: Evaluating top {config.top_k} alphas on test set ===")
    results = []
    backtests = {}
    for alpha_dict in evolved[:config.top_k]:
        expr_str = alpha_dict["expression"]
        tree = alpha_dict["tree"]
        alpha_values = _eval_expr(tree, pset, test.panels)

        if alpha_values.empty or alpha_values.isna().all().all():
            print(f"  Skipping (all NaN): {expr_str[:60]}")
            continue

        ic_series = compute_ic(alpha_values, test.forward_returns)
        ic_mean = float(ic_series.mean()) if not ic_series.empty else 0.0
        icir = compute_icir(ic_series)
        turnover = compute_turnover(alpha_values)
        bt = backtest_alpha(alpha_values, test.forward_returns)
        backtests[expr_str] = bt

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
        print(
            f"    Train IC: {alpha_dict['fitness']:.4f} | Test IC: {ic_mean:.4f} | "
            f"Sharpe: {bt.sharpe:.2f} | Return: {bt.annual_return:.2%}"
        )

    if backtests:
        plot_equity_curves(backtests, os.path.join(out_dir, "equity_curves.png"))

    report_path = os.path.join(out_dir, "alpha_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Alpha-GPT Final Report\n\n")
        f.write(f"**Trading Idea:** {trading_idea}\n\n")

        for i, result in enumerate(results):
            f.write(f"## Rank {i + 1}: `{result['expression']}`\n\n")
            f.write(f"- **Test IC:** {result['test_ic']:.4f}\n")
            f.write(f"- **ICIR:** {result['icir']:.4f}\n")
            f.write(f"- **Annual Return:** {result['annual_return']:.2%}\n")
            f.write(f"- **Sharpe:** {result['sharpe']:.2f}\n")
            f.write(f"- **Max Drawdown:** {result['max_drawdown']:.2%}\n\n")
            f.write("### AI Explanation\n")
            f.write(f"{result['explanation']}\n\n")
            f.write("---\n\n")

    print(f"\nSaved final report to {report_path}")
    return results


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run Alpha-GPT pipeline")
    parser.add_argument(
        "idea",
        nargs="*",
        help="Trading idea in natural language.",
    )
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Run only Stage 1 and Stage 2 debate, skipping GP and backtest.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    idea = " ".join(args.idea).strip() or DEFAULT_IDEA
    print(f"Trading idea: {idea}")
    if args.debate_only:
        run_debate_only(idea)
    else:
        run_pipeline(idea)
