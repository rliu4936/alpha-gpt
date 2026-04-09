"""Alpha-GPT 3.0: End-to-end pipeline orchestrator with 5 ablation modes.

Usage:
    python -m alpha_gpt.main "trading idea"                          # default: debate-only
    python -m alpha_gpt.main "trading idea" --mode full              # debate + GP
    python -m alpha_gpt.main "trading idea" --mode random-gp         # random GP (no LLM)
    python -m alpha_gpt.main "trading idea" --mode single-agent      # 1 LLM call, no GP
    python -m alpha_gpt.main "trading idea" --mode single-agent-gp   # 1 LLM call + GP
    python -m alpha_gpt.main "trading idea" --mode debate-only --num-runs 3
    python -m alpha_gpt.main --compare
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from openai import OpenAI

from alpha_gpt.analysis.explainer import explain_alpha
from alpha_gpt.analysis.metrics import compute_ic, compute_icir, compute_turnover
from alpha_gpt.analysis.visualize import (
    plot_comparison_curves,
    plot_equity_curves,
    plot_gp_evolution,
)
from alpha_gpt.backtest.backtester import backtest_alpha
from alpha_gpt.config import Config
from alpha_gpt.data.loader import DataSplit, load_panels, split_data
from alpha_gpt.debate.models import to_jsonable
from alpha_gpt.debate.moderator import run_formula_debate, run_idea_debate
from alpha_gpt.debate.prompts import SINGLE_AGENT_FORMULA_USER, SINGLE_AGENT_SYSTEM
from alpha_gpt.gp_search.engine import _eval_expr, run_gp
from alpha_gpt.gp_search.primitives import DEFAULT_TERMINALS, create_primitive_set
from alpha_gpt.gp_search.seed_injector import inject_seeds, normalize_expression, parse_expression

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
DEFAULT_IDEA = "Stocks with high momentum and low valuation tend to outperform"

VALID_MODES = ["random-gp", "single-agent", "single-agent-gp", "debate-only", "full"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_and_split(config: Config):
    """Load data panels and split into train/test.

    Returns:
        (train, test, available_terminals, pset)
    """
    panels = load_panels(config.data_dir)
    train, val, test = split_data(panels, config.train_end, config.val_end)
    del val  # reserved for future use
    available_terminals = [name for name in DEFAULT_TERMINALS if name in train.panels]
    pset = create_primitive_set(available_terminals)
    return train, test, available_terminals, pset


def _compute_vw_benchmark(test: DataSplit, cache_path: str = "data/panels/vw_market_return.parquet") -> pd.Series:
    """Compute value-weighted market return from test data.

    Uses market_cap weights if available, else equal-weight.
    Returns cumulative return series.
    """
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        # Filter to test period
        common = cached.index.intersection(test.forward_returns.index)
        if len(common) > 10:
            vw_ret = cached.loc[common].iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached.loc[common]
            return (1 + vw_ret).cumprod()

    fwd = test.forward_returns
    if "market_cap" in test.panels:
        mcap = test.panels["market_cap"].loc[fwd.index]
        weights = mcap.div(mcap.sum(axis=1), axis=0)
        vw_ret = (fwd * weights).sum(axis=1)
    else:
        # Equal-weight fallback
        vw_ret = fwd.mean(axis=1)

    # Cache it
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    vw_ret.to_frame("vw_return").to_parquet(cache_path)

    return (1 + vw_ret).cumprod()


def _eval_seed_formulas(
    seed_expressions: list[str],
    pset,
    test: DataSplit,
) -> list[dict]:
    """For no-GP modes: parse seeds, evaluate each on test data, return results."""
    results = []
    for expr_str in seed_expressions:
        tree = parse_expression(expr_str, pset)
        if tree is None:
            continue
        alpha_values = _eval_expr(tree, pset, test.panels)
        if alpha_values.empty or alpha_values.isna().all().all():
            continue

        ic_series = compute_ic(alpha_values, test.forward_returns)
        ic_mean = float(ic_series.mean()) if not ic_series.empty else 0.0

        results.append({
            "expression": normalize_expression(expr_str),
            "tree": tree,
            "fitness": ic_mean,
        })

    results.sort(key=lambda x: x["fitness"], reverse=True)
    return results


def _evaluate_and_report(
    alphas: list[dict],
    pset,
    test: DataSplit,
    client: OpenAI | None,
    config: Config,
    out_dir: str,
    trading_idea: str,
    benchmark: pd.Series | None,
    mode: str,
    run_idx: int,
    gp_seed: int | None = None,
) -> list[dict]:
    """Backtest top alphas, generate plots + report + results.json.

    Args:
        alphas: List of dicts with 'expression', 'tree', 'fitness'.
        pset: PrimitiveSet for eval.
        test: Test DataSplit.
        client: OpenAI client (may be None for random-gp).
        config: Config.
        out_dir: Output directory.
        trading_idea: The trading idea string.
        benchmark: VW benchmark cumulative return series.
        mode: Pipeline mode name.
        run_idx: Run index.
        gp_seed: GP random seed if applicable.

    Returns:
        List of result dicts.
    """
    results = []
    backtests = {}

    for alpha_dict in alphas[:config.top_k]:
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
        backtests[expr_str[:50]] = bt

        explanation = ""
        if client is not None:
            try:
                explanation = explain_alpha(
                    expression=expr_str,
                    backtest=bt,
                    ic_mean=ic_mean,
                    icir=icir,
                    client=client,
                    model=config.llm_model,
                )
            except Exception as e:
                logger.warning(f"Explanation failed: {e}")

        result = {
            "expression": expr_str,
            "train_ic": alpha_dict.get("fitness", 0.0),
            "test_ic": ic_mean,
            "icir": icir,
            "sharpe": bt.sharpe,
            "annual_return": bt.annual_return,
            "max_drawdown": bt.max_drawdown,
            "turnover": turnover,
            "explanation": explanation,
        }
        results.append(result)

        print(f"\n  Alpha: {expr_str[:70]}")
        print(
            f"    Test IC: {ic_mean:.4f} | ICIR: {icir:.4f} | "
            f"Sharpe: {bt.sharpe:.2f} | Return: {bt.annual_return:.2%}"
        )

    # Equity curves plot
    if backtests:
        plot_equity_curves(backtests, os.path.join(out_dir, "equity_curves.png"), benchmark=benchmark)

    # Markdown report
    report_path = os.path.join(out_dir, "alpha_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Alpha-GPT Report ({mode})\n\n")
        f.write(f"**Trading Idea:** {trading_idea}\n\n")
        f.write(f"**Mode:** `{mode}` | **Run:** {run_idx}\n\n")
        for i, r in enumerate(results):
            f.write(f"## Rank {i + 1}: `{r['expression']}`\n\n")
            f.write(f"- **Test IC:** {r['test_ic']:.4f}\n")
            f.write(f"- **ICIR:** {r['icir']:.4f}\n")
            f.write(f"- **Annual Return:** {r['annual_return']:.2%}\n")
            f.write(f"- **Sharpe:** {r['sharpe']:.2f}\n")
            f.write(f"- **Max Drawdown:** {r['max_drawdown']:.2%}\n\n")
            if r["explanation"]:
                f.write("### AI Explanation\n")
                f.write(f"{r['explanation']}\n\n")
            f.write("---\n\n")
    print(f"\nSaved report to {report_path}")

    # Machine-readable results.json
    top_alphas = [
        {
            "expression": r["expression"],
            "test_ic": r["test_ic"],
            "icir": r["icir"],
            "sharpe": r["sharpe"],
            "annual_return": r["annual_return"],
            "max_drawdown": r["max_drawdown"],
            "turnover": r["turnover"],
        }
        for r in results
    ]
    results_json = {
        "mode": mode,
        "run": run_idx,
        "seed": gp_seed,
        "trading_idea": trading_idea,
        "top_alphas": top_alphas,
    }
    results_json_path = os.path.join(out_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"Saved results.json to {results_json_path}")

    return results


# ---------------------------------------------------------------------------
# Single-agent LLM call
# ---------------------------------------------------------------------------

def _call_single_agent(trading_idea: str, available_terminals: list[str], config: Config) -> list[str]:
    """One LLM call to generate seed formulas directly (no debate)."""
    client = _create_client(config)
    user_msg = SINGLE_AGENT_FORMULA_USER.format(
        trading_idea=trading_idea,
        available_terminals=", ".join(available_terminals),
        target_count=config.seed_formula_target,
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": SINGLE_AGENT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=2500,
        temperature=0.4,
    )
    raw = response.choices[0].message.content.strip()

    # Extract JSON array from response
    from alpha_gpt.debate.agents import _extract_json_payload
    payload = _extract_json_payload(raw)
    if not isinstance(payload, list):
        logger.warning("Single-agent did not return a JSON array, attempting line parse")
        return []

    expressions = []
    for item in payload:
        if isinstance(item, dict) and item.get("expression"):
            expressions.append(item["expression"])
    return expressions


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _make_out_dir(mode: str, run_idx: int) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"{timestamp}_{mode}_run{run_idx}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_random_gp(trading_idea: str, config: Config, out_dir: str, gp_seed: int = 0):
    """Mode: random-gp. Pure random GP, no LLM involvement."""
    print(f"\n=== Mode: random-gp (seed={gp_seed}) ===")
    print(f"Output: {out_dir}")

    train, test, available_terminals, pset = _load_and_split(config)
    benchmark = _compute_vw_benchmark(test)

    print("\n--- GP search (random init) ---")
    evolved, logbook = run_gp(
        seed_trees=None,
        data=train,
        population_size=config.gp_population,
        generations=config.gp_generations,
        crossover_prob=config.gp_crossover,
        mutation_prob=config.gp_mutation,
        tournament_size=config.gp_tournament_size,
        max_depth=config.gp_max_depth,
        random_seed=gp_seed,
    )
    plot_gp_evolution(logbook, os.path.join(out_dir, "gp_evolution.png"))

    print(f"\n--- Evaluating top {config.top_k} alphas on test set ---")
    _evaluate_and_report(
        alphas=evolved,
        pset=pset,
        test=test,
        client=None,
        config=config,
        out_dir=out_dir,
        trading_idea=trading_idea,
        benchmark=benchmark,
        mode="random-gp",
        run_idx=0,
        gp_seed=gp_seed,
    )


def run_single_agent(trading_idea: str, config: Config, out_dir: str):
    """Mode: single-agent. One LLM call, direct formula eval, no GP."""
    print(f"\n=== Mode: single-agent ===")
    print(f"Output: {out_dir}")

    train, test, available_terminals, pset = _load_and_split(config)
    benchmark = _compute_vw_benchmark(test)

    print("\n--- Single-agent LLM call ---")
    seed_expressions = _call_single_agent(trading_idea, available_terminals, config)
    print(f"Got {len(seed_expressions)} formulas from single agent")

    # Save the raw formulas
    _save_json(os.path.join(out_dir, "single_agent_formulas.json"), seed_expressions)

    print(f"\n--- Evaluating seed formulas on test set ---")
    alphas = _eval_seed_formulas(seed_expressions, pset, test)
    print(f"Successfully evaluated {len(alphas)} formulas")

    _evaluate_and_report(
        alphas=alphas,
        pset=pset,
        test=test,
        client=_create_client(config),
        config=config,
        out_dir=out_dir,
        trading_idea=trading_idea,
        benchmark=benchmark,
        mode="single-agent",
        run_idx=0,
    )


def run_single_agent_gp(trading_idea: str, config: Config, out_dir: str, gp_seed: int = 0):
    """Mode: single-agent-gp. One LLM call to get seeds, then GP evolution."""
    print(f"\n=== Mode: single-agent-gp (seed={gp_seed}) ===")
    print(f"Output: {out_dir}")

    train, test, available_terminals, pset = _load_and_split(config)
    benchmark = _compute_vw_benchmark(test)

    print("\n--- Single-agent LLM call ---")
    seed_expressions = _call_single_agent(trading_idea, available_terminals, config)
    print(f"Got {len(seed_expressions)} formulas from single agent")

    _save_json(os.path.join(out_dir, "single_agent_formulas.json"), seed_expressions)

    seed_trees = inject_seeds(seed_expressions, pset)
    print(f"Parsed {len(seed_trees)}/{len(seed_expressions)} seeds into GP trees")

    print("\n--- GP search ---")
    evolved, logbook = run_gp(
        seed_trees=seed_trees,
        data=train,
        population_size=config.gp_population,
        generations=config.gp_generations,
        crossover_prob=config.gp_crossover,
        mutation_prob=config.gp_mutation,
        tournament_size=config.gp_tournament_size,
        max_depth=config.gp_max_depth,
        random_seed=gp_seed,
    )
    plot_gp_evolution(logbook, os.path.join(out_dir, "gp_evolution.png"))

    print(f"\n--- Evaluating top {config.top_k} alphas on test set ---")
    _evaluate_and_report(
        alphas=evolved,
        pset=pset,
        test=test,
        client=_create_client(config),
        config=config,
        out_dir=out_dir,
        trading_idea=trading_idea,
        benchmark=benchmark,
        mode="single-agent-gp",
        run_idx=0,
        gp_seed=gp_seed,
    )


def run_debate_only_mode(trading_idea: str, config: Config, out_dir: str):
    """Mode: debate-only. Multi-agent debate, direct formula eval, no GP."""
    print(f"\n=== Mode: debate-only ===")
    print(f"Output: {out_dir}")

    train, test, available_terminals, pset = _load_and_split(config)
    benchmark = _compute_vw_benchmark(test)
    client = _create_client(config)

    print("\n--- Stage 1: Idea debate ---")
    hypotheses, idea_artifacts = run_idea_debate(
        trading_idea=trading_idea,
        available_terminals=available_terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )
    print(f"Generated {len(hypotheses)} hypothesis specs")

    print("\n--- Stage 2: Formula debate ---")
    seed_pack, formula_artifacts = run_formula_debate(
        hypotheses=hypotheses,
        available_terminals=available_terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )
    print(f"Selected {len(seed_pack.selected_formulas)} seed formulas")

    if config.save_debate_artifacts:
        debate_artifacts = {}
        debate_artifacts.update(idea_artifacts)
        debate_artifacts.update(formula_artifacts)
        _save_debate_artifacts(os.path.join(out_dir, "debate"), debate_artifacts)

    seed_expressions = [f.expression for f in seed_pack.selected_formulas if f.expression]

    print(f"\n--- Evaluating seed formulas on test set ---")
    alphas = _eval_seed_formulas(seed_expressions, pset, test)
    print(f"Successfully evaluated {len(alphas)} formulas")

    _evaluate_and_report(
        alphas=alphas,
        pset=pset,
        test=test,
        client=client,
        config=config,
        out_dir=out_dir,
        trading_idea=trading_idea,
        benchmark=benchmark,
        mode="debate-only",
        run_idx=0,
    )


def run_full(trading_idea: str, config: Config, out_dir: str, gp_seed: int = 0):
    """Mode: full. Multi-agent debate + GP evolution."""
    print(f"\n=== Mode: full (seed={gp_seed}) ===")
    print(f"Output: {out_dir}")

    train, test, available_terminals, pset = _load_and_split(config)
    benchmark = _compute_vw_benchmark(test)
    client = _create_client(config)

    print("\n--- Stage 1: Idea debate ---")
    hypotheses, idea_artifacts = run_idea_debate(
        trading_idea=trading_idea,
        available_terminals=available_terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )

    print("\n--- Stage 2: Formula debate ---")
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

    seed_expressions = [f.expression for f in seed_pack.selected_formulas if f.expression]
    seed_trees = inject_seeds(seed_expressions, pset)
    print(f"Parsed {len(seed_trees)}/{len(seed_expressions)} seeds into GP trees")

    print("\n--- GP search ---")
    evolved, logbook = run_gp(
        seed_trees=seed_trees,
        data=train,
        population_size=config.gp_population,
        generations=config.gp_generations,
        crossover_prob=config.gp_crossover,
        mutation_prob=config.gp_mutation,
        tournament_size=config.gp_tournament_size,
        max_depth=config.gp_max_depth,
        random_seed=gp_seed,
    )
    plot_gp_evolution(logbook, os.path.join(out_dir, "gp_evolution.png"))

    print(f"\n--- Evaluating top {config.top_k} alphas on test set ---")
    _evaluate_and_report(
        alphas=evolved,
        pset=pset,
        test=test,
        client=client,
        config=config,
        out_dir=out_dir,
        trading_idea=trading_idea,
        benchmark=benchmark,
        mode="full",
        run_idx=0,
        gp_seed=gp_seed,
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def run_compare():
    """Scan outputs/*/results.json and produce cross-mode comparison report."""
    import glob
    from collections import defaultdict

    pattern = os.path.join("outputs", "*", "results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print("No results.json files found in outputs/")
        return

    all_results = []
    for fpath in files:
        with open(fpath) as f:
            all_results.append(json.load(f))

    # Group by mode
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_mode[r["mode"]].append(r)

    compare_dir = os.path.join("outputs", "comparison")
    os.makedirs(compare_dir, exist_ok=True)

    # Build summary rows — two aggregation levels per mode
    rows = []
    for mode, runs in sorted(by_mode.items()):
        median_ics, median_icirs, median_sharpes = [], [], []
        best_ics, best_icirs, best_sharpes = [], [], []

        for run in runs:
            alphas = run.get("top_alphas", [])
            if not alphas:
                continue

            ics = [a["test_ic"] for a in alphas]
            icirs = [a["icir"] for a in alphas]
            sharpes = [a["sharpe"] for a in alphas]

            # Median of top-k for this run
            median_ics.append(float(np.median(ics)))
            median_icirs.append(float(np.median(icirs)))
            median_sharpes.append(float(np.median(sharpes)))

            # Best single alpha in this run
            best_idx = int(np.argmax(ics))
            best_ics.append(ics[best_idx])
            best_icirs.append(icirs[best_idx])
            best_sharpes.append(sharpes[best_idx])

        if not median_ics:
            continue

        rows.append({
            "mode": mode,
            "n_runs": len(runs),
            # Median of top-k, aggregated across runs
            "median_ic_mean": float(np.mean(median_ics)),
            "median_ic_std": float(np.std(median_ics)),
            "median_icir_mean": float(np.mean(median_icirs)),
            "median_icir_std": float(np.std(median_icirs)),
            "median_sharpe_mean": float(np.mean(median_sharpes)),
            "median_sharpe_std": float(np.std(median_sharpes)),
            # Best alpha, aggregated across runs
            "best_ic_mean": float(np.mean(best_ics)),
            "best_ic_std": float(np.std(best_ics)),
            "best_icir_mean": float(np.mean(best_icirs)),
            "best_icir_std": float(np.std(best_icirs)),
            "best_sharpe_mean": float(np.mean(best_sharpes)),
            "best_sharpe_std": float(np.std(best_sharpes)),
        })

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY — Median of Top-K Alphas")
    print("=" * 80)
    print(f"{'Mode':<18} {'Runs':>4}  {'IC':>18}  {'ICIR':>18}  {'Sharpe':>18}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['mode']:<18} {row['n_runs']:>4}  "
            f"{row['median_ic_mean']:>7.4f}+/-{row['median_ic_std']:<7.4f}  "
            f"{row['median_icir_mean']:>7.4f}+/-{row['median_icir_std']:<7.4f}  "
            f"{row['median_sharpe_mean']:>7.2f}+/-{row['median_sharpe_std']:<7.2f}"
        )

    print()
    print("COMPARISON SUMMARY — Best Alpha per Run")
    print("=" * 80)
    print(f"{'Mode':<18} {'Runs':>4}  {'IC':>18}  {'ICIR':>18}  {'Sharpe':>18}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['mode']:<18} {row['n_runs']:>4}  "
            f"{row['best_ic_mean']:>7.4f}+/-{row['best_ic_std']:<7.4f}  "
            f"{row['best_icir_mean']:>7.4f}+/-{row['best_icir_std']:<7.4f}  "
            f"{row['best_sharpe_mean']:>7.2f}+/-{row['best_sharpe_std']:<7.2f}"
        )

    # --- Markdown report ---
    table_path = os.path.join(compare_dir, "summary_table.md")
    with open(table_path, "w") as f:
        f.write("# Cross-Mode Comparison\n\n")

        f.write("## Median of Top-K Alphas\n\n")
        f.write("| Mode | Runs | IC (mean +/- std) | ICIR (mean +/- std) | Sharpe (mean +/- std) |\n")
        f.write("|------|------|--------------------|---------------------|-----------------------|\n")
        for row in rows:
            f.write(
                f"| {row['mode']} | {row['n_runs']} | "
                f"{row['median_ic_mean']:.4f} +/- {row['median_ic_std']:.4f} | "
                f"{row['median_icir_mean']:.4f} +/- {row['median_icir_std']:.4f} | "
                f"{row['median_sharpe_mean']:.2f} +/- {row['median_sharpe_std']:.2f} |\n"
            )

        f.write("\n## Best Alpha per Run\n\n")
        f.write("| Mode | Runs | IC (mean +/- std) | ICIR (mean +/- std) | Sharpe (mean +/- std) |\n")
        f.write("|------|------|--------------------|---------------------|-----------------------|\n")
        for row in rows:
            f.write(
                f"| {row['mode']} | {row['n_runs']} | "
                f"{row['best_ic_mean']:.4f} +/- {row['best_ic_std']:.4f} | "
                f"{row['best_icir_mean']:.4f} +/- {row['best_icir_std']:.4f} | "
                f"{row['best_sharpe_mean']:.2f} +/- {row['best_sharpe_std']:.2f} |\n"
            )
    print(f"\nSaved summary table to {table_path}")

    # --- JSON ---
    comparison_json_path = os.path.join(compare_dir, "comparison.json")
    with open(comparison_json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved comparison.json to {comparison_json_path}")

    print(f"\nComparison complete. See {compare_dir}/")


# ---------------------------------------------------------------------------
# Legacy wrappers (backward compat)
# ---------------------------------------------------------------------------

def run_debate_only(
    trading_idea: str,
    config: Config | None = None,
    available_terminals: list[str] | None = None,
) -> list[dict[str, str]]:
    """Legacy wrapper: run debate-only and return seed formula dicts."""
    config = config or Config()
    terminals = available_terminals or list(DEFAULT_TERMINALS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"{timestamp}_debate-only_run0")
    os.makedirs(out_dir, exist_ok=True)

    client = _create_client(config)

    hypotheses, idea_artifacts = run_idea_debate(
        trading_idea=trading_idea,
        available_terminals=terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )

    seed_pack, formula_artifacts = run_formula_debate(
        hypotheses=hypotheses,
        available_terminals=terminals,
        client=client,
        model=config.llm_model,
        config=config,
    )

    if config.save_debate_artifacts:
        debate_artifacts = {}
        debate_artifacts.update(idea_artifacts)
        debate_artifacts.update(formula_artifacts)
        _save_debate_artifacts(os.path.join(out_dir, "debate"), debate_artifacts)

    return [
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


def run_pipeline(trading_idea: str, config: Config | None = None) -> list[dict]:
    """Legacy wrapper: run full pipeline (debate + GP)."""
    config = config or Config()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", f"{timestamp}_full_run0")
    os.makedirs(out_dir, exist_ok=True)
    run_full(trading_idea, config, out_dir, gp_seed=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for pipeline execution."""
    parser = argparse.ArgumentParser(description="Run Alpha-GPT pipeline")
    parser.add_argument(
        "idea",
        nargs="*",
        help="Trading idea in natural language.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="debate-only",
        help="Pipeline mode (default: debate-only).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs with GP seeds 0..N-1 (default: 1).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Aggregate existing results across modes and produce comparison report.",
    )
    # Legacy flag
    parser.add_argument(
        "--debate-only",
        action="store_true",
        help="Legacy alias for --mode debate-only.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if args.compare:
        run_compare()
        sys.exit(0)

    idea = " ".join(args.idea).strip() or DEFAULT_IDEA
    mode = args.mode
    if args.debate_only:
        mode = "debate-only"

    config = Config()
    print(f"Trading idea: {idea}")
    print(f"Mode: {mode} | Runs: {args.num_runs}")

    for run_idx in range(args.num_runs):
        out_dir = _make_out_dir(mode, run_idx)
        match mode:
            case "random-gp":
                run_random_gp(idea, config, out_dir, gp_seed=run_idx)
            case "single-agent":
                run_single_agent(idea, config, out_dir)
            case "single-agent-gp":
                run_single_agent_gp(idea, config, out_dir, gp_seed=run_idx)
            case "debate-only":
                run_debate_only_mode(idea, config, out_dir)
            case "full":
                run_full(idea, config, out_dir, gp_seed=run_idx)
