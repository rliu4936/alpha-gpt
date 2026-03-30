"""Parse LLM-generated alpha expression strings into DEAP PrimitiveTree objects.

Handles the bridge between natural-language-style expressions like
  cs_rank(ts_delta(close, 5))
and DEAP's curried form:
  cs_rank(ts_delta_5(close))
"""

import re
import logging

from deap import gp

logger = logging.getLogger(__name__)

# Mapping from generic operator + window to curried name
CURRIED_MAP = {}
for op in ["ts_mean", "ts_std", "ts_min", "ts_max"]:
    for w in [5, 10, 20, 60]:
        CURRIED_MAP[(op, w)] = f"{op}_{w}"
for op in ["ts_delta", "ts_rank"]:
    for w in [5, 10, 20]:
        CURRIED_MAP[(op, w)] = f"{op}_{w}"
for w in [1, 5, 20]:
    CURRIED_MAP[("ts_returns", w)] = f"ts_returns_{w}"
for w in [10, 20]:
    CURRIED_MAP[("ts_corr", w)] = f"ts_corr_{w}"

# Default window for each operator family (used when LLM omits the window)
DEFAULT_WINDOWS = {
    "ts_mean": 20, "ts_std": 20, "ts_min": 20, "ts_max": 20,
    "ts_delta": 5, "ts_rank": 10, "ts_returns": 1, "ts_corr": 20,
}

# Operators that are already curried (no window parameter)
NO_WINDOW_OPS = {
    "cs_rank", "cs_zscore", "log_abs", "abs_val", "sign", "neg",
    "add", "sub", "mul", "safe_div",
}


def normalize_expression(expr: str) -> str:
    """Convert LLM-style expression to DEAP-compatible curried form.

    Examples:
        "cs_rank(ts_delta(close, 5))" -> "cs_rank(ts_delta_5(close))"
        "ts_mean(volume, 20)"        -> "ts_mean_20(volume)"
        "rank(returns)"              -> "cs_rank(returns)"
    """
    # Common LLM aliases
    expr = expr.replace("rank(", "cs_rank(")
    expr = expr.replace("zscore(", "cs_zscore(")
    expr = expr.replace("delta(", "ts_delta(")
    expr = expr.replace("std(", "ts_std(")
    expr = expr.replace("mean(", "ts_mean(")
    expr = expr.replace("abs(", "abs_val(")
    expr = expr.replace("log(", "log_abs(")
    expr = expr.replace("div(", "safe_div(")
    expr = expr.replace("correlation(", "ts_corr_20(")
    expr = expr.replace("corr(", "ts_corr_20(")

    # Handle ts_op(field, window) -> ts_op_window(field)
    pattern = r'(ts_\w+)\(([^,]+),\s*(\d+)\)'
    def replace_with_curried(match):
        op_name = match.group(1)
        field = match.group(2)
        window = int(match.group(3))
        curried = CURRIED_MAP.get((op_name, window))
        if curried:
            return f"{curried}({field})"
        # Try with default window
        default_w = DEFAULT_WINDOWS.get(op_name)
        if default_w:
            curried = CURRIED_MAP.get((op_name, default_w))
            if curried:
                return f"{curried}({field})"
        return match.group(0)  # leave unchanged

    # Apply repeatedly (nested expressions)
    for _ in range(5):
        new_expr = re.sub(pattern, replace_with_curried, expr)
        if new_expr == expr:
            break
        expr = new_expr

    # Handle ts_op(field) without window -> add default window
    for op_name, default_w in DEFAULT_WINDOWS.items():
        # Match op_name( but not op_name_\d+( (already curried)
        pattern_no_window = rf'(?<!\w){op_name}\((?!\d)'
        curried = CURRIED_MAP.get((op_name, default_w), f"{op_name}_{default_w}")
        expr = re.sub(pattern_no_window, f"{curried}(", expr)

    return expr


def parse_expression(expr_str: str, pset: gp.PrimitiveSet) -> gp.PrimitiveTree | None:
    """Parse an expression string into a DEAP PrimitiveTree.

    Args:
        expr_str: Expression string (LLM-generated or normalized).
        pset: The PrimitiveSet to use for parsing.

    Returns:
        PrimitiveTree if parsing succeeds, None if it fails.
    """
    try:
        normalized = normalize_expression(expr_str)
        tree = gp.PrimitiveTree.from_string(normalized, pset)
        return tree
    except Exception as e:
        logger.warning(f"Failed to parse expression: {expr_str!r} -> {e}")
        return None


def inject_seeds(
    seed_expressions: list[str],
    pset: gp.PrimitiveSet,
) -> list[gp.PrimitiveTree]:
    """Parse a list of LLM-generated expressions into DEAP trees.

    Expressions that fail to parse are logged and skipped.

    Returns:
        List of successfully parsed PrimitiveTree objects.
    """
    trees = []
    for expr in seed_expressions:
        tree = parse_expression(expr, pset)
        if tree is not None:
            trees.append(tree)
            logger.info(f"Parsed seed: {expr!r} -> {tree}")
        else:
            logger.warning(f"Skipped unparseable seed: {expr!r}")

    logger.info(f"Injected {len(trees)}/{len(seed_expressions)} seed expressions")
    return trees
