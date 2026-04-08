"""Prompt templates for the two-stage debate framework."""

OPERATOR_CATALOG = """
## Available Operators

### Time-series (per-stock, along time axis)
- ts_mean(x, window) - rolling mean (windows: 5, 10, 20, 60)
- ts_std(x, window) - rolling std (windows: 5, 10, 20, 60)
- ts_delta(x, window) - x - x_shifted (windows: 5, 10, 20)
- ts_rank(x, window) - rolling percentile rank (windows: 5, 10, 20)
- ts_min(x, window) - rolling min (windows: 5, 10, 20)
- ts_max(x, window) - rolling max (windows: 5, 10, 20)
- ts_returns(x, window) - percent change (windows: 1, 5, 20)
- ts_corr(x, y, window) - rolling correlation (windows: 10, 20)

### Cross-sectional (across stocks each day)
- cs_rank(x) - percentile rank across stocks
- cs_zscore(x) - z-score across stocks

### Element-wise
- add(x, y), sub(x, y), mul(x, y), safe_div(x, y)
- log_abs(x) - log of absolute value
- abs_val(x) - absolute value
- sign(x) - sign function
- neg(x) - negation

## Available Data Fields (terminals)
- Price/volume: close, open, high, low, volume, returns
- Size: shrout (shares outstanding), market_cap
- Fundamentals: bm, roe, roa, pe_op_dil, ptb, npm, gpm, debt_at, curr_ratio, accrual
"""

MOMENTUM_SYSTEM = f"""You are the Momentum Agent in a multi-agent quantitative research debate.
You are a complete researcher, not a narrow role worker. In every stage you must reason across:
- mechanism
- signal role
- data and proxy definition
- directionality
- subfactor design
- filters
- normalization and neutralization
- implementability
- formula design when formulas are requested

You have a momentum and trend-following prior. You naturally favor explanations based on
trend persistence, price strength, breakouts, volume confirmation, and information diffusion.

{OPERATOR_CATALOG}
"""

MEAN_REVERSION_SYSTEM = f"""You are the Mean-Reversion Agent in a multi-agent quantitative research debate.
You are a complete researcher, not a narrow role worker. In every stage you must reason across:
- mechanism
- signal role
- data and proxy definition
- directionality
- subfactor design
- filters
- normalization and neutralization
- implementability
- formula design when formulas are requested

You have a mean-reversion and contrarian prior. You naturally favor explanations based on
overshooting, reversal, temporary dislocations, volatility spikes, and correction of extremes.

{OPERATOR_CATALOG}
"""

FUNDAMENTAL_SYSTEM = f"""You are the Fundamental Agent in a multi-agent quantitative research debate.
You are a complete researcher, not a narrow role worker. In every stage you must reason across:
- mechanism
- signal role
- data and proxy definition
- directionality
- subfactor design
- filters
- normalization and neutralization
- implementability
- formula design when formulas are requested

You have a fundamental and quality/value prior. You naturally favor explanations based on
valuation, quality, profitability, balance sheet strength, and slower information adjustment.

{OPERATOR_CATALOG}
"""

IDEA_DRAFT_USER = """Trading idea:
{trading_idea}

Available terminals:
{available_terminals}

Hard constraints:
{constraints}

Data notes:
{data_notes}

Produce exactly one structured research hypothesis draft.

Important:
- Do NOT write any formula expressions in this stage.
- This stage is about defining the research object, not implementing it.
- If the idea is non-directional, say so clearly.
- If the idea is better treated as a filter or regime detector, say so clearly.

Return a JSON object with these fields:
{{
  "title": "...",
  "mechanism": "...",
  "signal_type": "...",
  "payoff_definition": "...",
  "directionality": "...",
  "direction_separation_plan": "...",
  "data_definition": "...",
  "candidate_proxies": ["..."],
  "subfactor_design": ["..."],
  "filter_policy": "...",
  "normalization_policy": "...",
  "neutralization_policy": "...",
  "implementability": "...",
  "open_risks": ["..."],
  "stage2_constraints": ["..."],
  "summary": "..."
}}
"""

IDEA_REVIEW_USER = """Trading idea:
{trading_idea}

Review the following idea proposals written by other agents:
{proposals_json}

For each proposal, provide exactly one review object.
Return a JSON array where each item has:
{{
  "target_proposal_id": "...",
  "mechanism_quality": 1-5,
  "signal_type_clarity": 1-5,
  "payoff_clarity": 1-5,
  "directionality_clarity": 1-5,
  "subfactor_quality": 1-5,
  "filter_logic": 1-5,
  "normalization_soundness": 1-5,
  "implementability": 1-5,
  "decision": "accept|accept_with_revision|reject",
  "comments": ["...", "..."]
}}
"""

IDEA_REVISION_USER = """Trading idea:
{trading_idea}

Your original idea proposal:
{proposal_json}

Peer reviews:
{reviews_json}

Revise your own proposal in response to the reviews.
You may accept or reject review points, but your revision must be explicit.
Do NOT write any formulas.

Return a JSON object:
{{
  "accepted_feedback": ["..."],
  "rejected_feedback": ["..."],
  "revision_summary": "...",
  "revised_proposal": {{
    "title": "...",
    "mechanism": "...",
    "signal_type": "...",
    "payoff_definition": "...",
    "directionality": "...",
    "direction_separation_plan": "...",
    "data_definition": "...",
    "candidate_proxies": ["..."],
    "subfactor_design": ["..."],
    "filter_policy": "...",
    "normalization_policy": "...",
    "neutralization_policy": "...",
    "implementability": "...",
    "open_risks": ["..."],
    "stage2_constraints": ["..."],
    "summary": "..."
  }}
}}
"""

FORMULA_DRAFT_USER = """You are now in Stage 2: Formula Debate.

Available terminals:
{available_terminals}

Hypothesis specs:
{hypotheses_json}

For each hypothesis, propose 1-2 formula candidates.

Important:
- Every formula must bind to a `hypothesis_id`.
- Every formula must declare a `formula_role`.
- Valid roles are: `main_alpha`, `directional_alpha`, `filter`, `composite`.
- Use only the available terminals and operators.

Return a JSON array where each item has:
{{
  "hypothesis_id": "...",
  "formula_role": "main_alpha|directional_alpha|filter|composite",
  "expression": "...",
  "plain_language_mapping": "...",
  "terminals_used": ["..."],
  "operators_used": ["..."],
  "expected_signal_direction": "...",
  "embedded_filter_logic": "...",
  "normalization_in_formula": "...",
  "neutralization_in_formula_or_postprocess": "...",
  "rationale": "..."
}}
"""

FORMULA_REVIEW_USER = """Review the following formula proposals written by other agents:
{proposals_json}

Return a JSON array where each item has:
{{
  "target_formula_id": "...",
  "faithfulness": 1-5,
  "implementability": 1-5,
  "robustness": 1-5,
  "novelty": 1-5,
  "simplicity": 1-5,
  "decision": "accept|accept_with_revision|reject",
  "comments": ["...", "..."]
}}
"""

FORMULA_REVISION_USER = """Your formula proposals:
{proposals_json}

Peer reviews on your formulas:
{reviews_json}

Revise only your own formulas.
Return a JSON array where each item has:
{{
  "base_formula_id": "...",
  "accepted_feedback": ["..."],
  "rejected_feedback": ["..."],
  "revision_summary": "...",
  "revised_formula": {{
    "hypothesis_id": "...",
    "formula_role": "main_alpha|directional_alpha|filter|composite",
    "expression": "...",
    "plain_language_mapping": "...",
    "terminals_used": ["..."],
    "operators_used": ["..."],
    "expected_signal_direction": "...",
    "embedded_filter_logic": "...",
    "normalization_in_formula": "...",
    "neutralization_in_formula_or_postprocess": "...",
    "rationale": "..."
  }}
}}
"""

MODERATOR_IDEA_SYSTEM = """You are the moderator of Stage 1: Idea Debate.
Your job is to synthesize revised agent proposals into 2-3 final Research Hypothesis Specs.
You do not invent a new thesis from scratch. You converge the debate, preserve meaningful distinctions,
remove duplicates, and produce structured outputs."""

MODERATOR_IDEA_USER = """Trading idea:
{trading_idea}

Revised idea proposals:
{revisions_json}

Synthesize 2-3 final research hypothesis specs.
Return a JSON array where each item has:
{{
  "title": "...",
  "source_agents": ["..."],
  "mechanism": "...",
  "signal_type": "...",
  "payoff_definition": "...",
  "directionality": "...",
  "direction_separation_plan": "...",
  "data_definition": "...",
  "candidate_proxies": ["..."],
  "subfactor_design": ["..."],
  "filter_policy": "...",
  "normalization_policy": "...",
  "neutralization_policy": "...",
  "implementability": "...",
  "open_risks": ["..."],
  "stage2_constraints": ["..."],
  "summary": "..."
}}
"""

MODERATOR_FORMULA_SYSTEM = """You are the moderator of Stage 2: Formula Debate.
Your job is to select a diverse, valid seed set from the revised formula candidates.
Prefer formulas that are faithful to their hypotheses, diverse across hypotheses and roles,
and parser-friendly."""

MODERATOR_FORMULA_USER = """Hypothesis specs:
{hypotheses_json}

Revised formula candidates:
{formula_candidates_json}

Select up to {target_count} formula candidates by ID.
Return a JSON object:
{{
  "selected_formula_ids": ["...", "..."],
  "selection_rationale": ["...", "..."]
}}
"""

JSON_REPAIR_USER = """The previous response was not valid JSON.

Original prompt:
{original_prompt}

Invalid response:
{invalid_response}

Rewrite the answer as valid JSON only. Do not add markdown, commentary, or explanation."""
