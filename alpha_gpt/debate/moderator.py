"""Two-stage debate orchestration and moderator synthesis."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from alpha_gpt.config import Config
from alpha_gpt.debate.agents import _extract_json_payload, create_agents
from alpha_gpt.debate.models import (
    FormulaDebateBrief,
    FormulaProposal,
    FormulaRevision,
    FormulaReview,
    IdeaDebateBrief,
    IdeaProposal,
    IdeaRevision,
    IdeaReview,
    ResearchHypothesisSpec,
    SeedFormulaPack,
    make_id,
    to_jsonable,
)
from alpha_gpt.debate.prompts import (
    MODERATOR_FORMULA_SYSTEM,
    MODERATOR_FORMULA_USER,
    MODERATOR_IDEA_SYSTEM,
    MODERATOR_IDEA_USER,
    OPERATOR_CATALOG,
)

logger = logging.getLogger(__name__)


def _config_value(config: Config | None, name: str, default: Any) -> Any:
    return getattr(config, name, default) if config is not None else default


def _call_moderator_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    empty_factory,
    retries: int = 2,
) -> Any:
    """Call the moderator and parse JSON, with one repair retry."""
    invalid_response = ""
    retries = max(1, retries)
    for attempt in range(retries):
        if attempt == 0:
            prompt = user_prompt
        else:
            prompt = (
                "The previous answer was not valid JSON. "
                "Rewrite it as valid JSON only.\n\n"
                f"Original prompt:\n{user_prompt}\n\n"
                f"Invalid response:\n{invalid_response}"
            )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2500,
                temperature=0.2 if attempt == 0 else 0.0,
            )
            invalid_response = response.choices[0].message.content.strip()
            payload = _extract_json_payload(invalid_response)
            if payload is not None:
                return payload
        except Exception as exc:
            logger.error("Moderator call failed on attempt %s: %s", attempt + 1, exc)
            break

    return empty_factory()


def _coerce_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_str(item) for item in value if _coerce_str(item)]
    text = _coerce_str(value)
    return [text] if text else []


def _proposal_to_hypothesis(proposal: IdeaProposal, hypothesis_id: str | None = None) -> ResearchHypothesisSpec:
    return ResearchHypothesisSpec(
        hypothesis_id=hypothesis_id or make_id("hypothesis", proposal.agent_name, proposal.proposal_id),
        title=proposal.title,
        source_agents=[proposal.agent_name],
        mechanism=proposal.mechanism,
        signal_type=proposal.signal_type,
        payoff_definition=proposal.payoff_definition,
        directionality=proposal.directionality,
        direction_separation_plan=proposal.direction_separation_plan,
        data_definition=proposal.data_definition,
        candidate_proxies=list(proposal.candidate_proxies),
        subfactor_design=list(proposal.subfactor_design),
        filter_policy=proposal.filter_policy,
        normalization_policy=proposal.normalization_policy,
        neutralization_policy=proposal.neutralization_policy,
        implementability=proposal.implementability,
        open_risks=list(proposal.open_risks),
        stage2_constraints=list(proposal.stage2_constraints),
        summary=proposal.summary,
    )


def _hypothesis_from_payload(payload: Any, idx: int) -> ResearchHypothesisSpec:
    payload = payload if isinstance(payload, dict) else {}
    return ResearchHypothesisSpec(
        hypothesis_id=make_id("hypothesis", _coerce_str(payload.get("title")) or str(idx + 1)),
        title=_coerce_str(payload.get("title")) or f"Hypothesis {idx + 1}",
        source_agents=_coerce_list_of_str(payload.get("source_agents")),
        mechanism=_coerce_str(payload.get("mechanism")),
        signal_type=_coerce_str(payload.get("signal_type")),
        payoff_definition=_coerce_str(payload.get("payoff_definition")),
        directionality=_coerce_str(payload.get("directionality")),
        direction_separation_plan=_coerce_str(payload.get("direction_separation_plan")),
        data_definition=_coerce_str(payload.get("data_definition")),
        candidate_proxies=_coerce_list_of_str(payload.get("candidate_proxies")),
        subfactor_design=_coerce_list_of_str(payload.get("subfactor_design")),
        filter_policy=_coerce_str(payload.get("filter_policy")),
        normalization_policy=_coerce_str(payload.get("normalization_policy")),
        neutralization_policy=_coerce_str(payload.get("neutralization_policy")),
        implementability=_coerce_str(payload.get("implementability")),
        open_risks=_coerce_list_of_str(payload.get("open_risks")),
        stage2_constraints=_coerce_list_of_str(payload.get("stage2_constraints")),
        summary=_coerce_str(payload.get("summary")),
    )


def _build_idea_brief(trading_idea: str, available_terminals: list[str]) -> IdeaDebateBrief:
    constraints = [
        "Do not use future information or look-ahead logic.",
        "Do not write formulas during Stage 1.",
        "Use only concepts that can be supported directly or through proxies from the available terminals.",
    ]
    data_notes = [
        f"Available terminals for this run: {', '.join(available_terminals) or 'none'}.",
        "Cross-sectional and time-series roles should be made explicit in the hypothesis.",
    ]
    return IdeaDebateBrief(
        trading_idea=trading_idea,
        available_terminals=available_terminals,
        operator_catalog=OPERATOR_CATALOG,
        constraints=constraints,
        data_notes=data_notes,
    )


def _build_formula_briefs(
    hypotheses: list[ResearchHypothesisSpec],
    available_terminals: list[str],
) -> list[FormulaDebateBrief]:
    briefs = []
    for hypothesis in hypotheses:
        constraints = list(hypothesis.stage2_constraints)
        constraints.append(f"Use only these terminals: {', '.join(available_terminals) or 'none'}.")
        briefs.append(
            FormulaDebateBrief(
                hypothesis_id=hypothesis.hypothesis_id,
                hypothesis_title=hypothesis.title,
                hypothesis_summary=hypothesis.summary,
                available_terminals=available_terminals,
                operator_catalog=OPERATOR_CATALOG,
                formula_constraints=constraints,
            )
        )
    return briefs


def _annotate_formula_parse_status(formulas: list[FormulaProposal], available_terminals: list[str]) -> None:
    try:
        from alpha_gpt.gp_search.primitives import create_primitive_set
        from alpha_gpt.gp_search.seed_injector import normalize_expression, parse_expression
    except ModuleNotFoundError:
        for formula in formulas:
            formula.normalized_expression = formula.expression
            formula.parseable = bool(formula.expression) and "not_a_formula" not in formula.expression
            formula.parse_error = "" if formula.parseable else "Parser dependencies are unavailable."
        return

    if not available_terminals:
        for formula in formulas:
            formula.parseable = False
            formula.parse_error = "No available terminals for parse gate."
            formula.normalized_expression = formula.expression
        return

    pset = create_primitive_set(available_terminals)
    for formula in formulas:
        if not formula.expression:
            formula.parseable = False
            formula.parse_error = "Empty expression."
            formula.normalized_expression = ""
            continue
        normalized = normalize_expression(formula.expression)
        formula.normalized_expression = normalized
        tree = parse_expression(formula.expression, pset)
        if tree is None:
            formula.parseable = False
            formula.parse_error = "Parser returned None."
        else:
            formula.parseable = True
            formula.parse_error = ""


def _synthesize_hypotheses(
    trading_idea: str,
    revisions: list[IdeaRevision],
    client: OpenAI,
    model: str,
    target_count: int,
    retries: int,
) -> list[ResearchHypothesisSpec]:
    user_msg = MODERATOR_IDEA_USER.format(
        trading_idea=trading_idea,
        revisions_json=json.dumps(to_jsonable(revisions), indent=2, ensure_ascii=False),
    )
    payload = _call_moderator_json(
        client=client,
        model=model,
        system_prompt=MODERATOR_IDEA_SYSTEM,
        user_prompt=user_msg,
        empty_factory=list,
        retries=retries,
    )
    items = payload if isinstance(payload, list) else []
    hypotheses = [
        _hypothesis_from_payload(item, idx)
        for idx, item in enumerate(items[:target_count])
    ]
    if hypotheses:
        return hypotheses

    fallback = []
    for idx, revision in enumerate(revisions[:target_count]):
        fallback.append(
            _proposal_to_hypothesis(
                revision.revised_proposal,
                hypothesis_id=make_id("hypothesis", revision.agent_name, str(idx + 1)),
            )
        )
    return fallback


def _select_seed_formulas(
    hypotheses: list[ResearchHypothesisSpec],
    formulas: list[FormulaProposal],
    client: OpenAI,
    model: str,
    target_count: int,
    retries: int,
) -> SeedFormulaPack:
    parseable = [formula for formula in formulas if formula.parseable]
    dropped = [formula for formula in formulas if not formula.parseable]
    if not parseable:
        return SeedFormulaPack(
            pack_id=make_id("seed-pack", "empty"),
            selected_formulas=[],
            selection_rationale=["No parseable formulas were available after Stage 2."],
            traceability_map={},
            dropped_formulas=dropped,
        )

    user_msg = MODERATOR_FORMULA_USER.format(
        hypotheses_json=json.dumps(to_jsonable(hypotheses), indent=2, ensure_ascii=False),
        formula_candidates_json=json.dumps(to_jsonable(parseable), indent=2, ensure_ascii=False),
        target_count=target_count,
    )
    payload = _call_moderator_json(
        client=client,
        model=model,
        system_prompt=MODERATOR_FORMULA_SYSTEM,
        user_prompt=user_msg,
        empty_factory=dict,
        retries=retries,
    )

    parseable_by_id = {formula.formula_id: formula for formula in parseable}
    selected_ids = []
    selection_rationale = []
    if isinstance(payload, dict):
        selected_ids = [
            formula_id
            for formula_id in _coerce_list_of_str(payload.get("selected_formula_ids"))
            if formula_id in parseable_by_id
        ]
        selection_rationale = _coerce_list_of_str(payload.get("selection_rationale"))
        selected_ids = list(dict.fromkeys(selected_ids))

    if not selected_ids:
        selected_ids = [formula.formula_id for formula in parseable[:target_count]]
        selection_rationale = ["Fallback selection: first parseable formulas by revised order."]

    selected = [parseable_by_id[formula_id] for formula_id in selected_ids[:target_count]]
    selected_id_set = {formula.formula_id for formula in selected}
    dropped.extend(formula for formula in parseable if formula.formula_id not in selected_id_set)
    return SeedFormulaPack(
        pack_id=make_id("seed-pack", "selected"),
        selected_formulas=selected,
        selection_rationale=selection_rationale,
        traceability_map={formula.formula_id: formula.hypothesis_id for formula in selected},
        dropped_formulas=dropped,
    )


def run_idea_debate(
    trading_idea: str,
    available_terminals: list[str],
    client: OpenAI,
    model: str,
    config: Config | None = None,
) -> tuple[list[ResearchHypothesisSpec], dict[str, Any]]:
    """Run Stage 1 idea debate and return hypotheses plus saved artifacts."""
    brief = _build_idea_brief(trading_idea, available_terminals)
    json_retries = _config_value(config, "debate_json_retries", 2)
    target_count = _config_value(config, "idea_hypothesis_target", 3)
    agents = create_agents(client, model, json_retries=json_retries)

    idea_drafts = [agent.draft_idea(brief) for agent in agents]

    idea_reviews: list[IdeaReview] = []
    for agent in agents:
        targets = [proposal for proposal in idea_drafts if proposal.agent_name != agent.name]
        idea_reviews.extend(agent.review_ideas(brief, targets))

    idea_revisions: list[IdeaRevision] = []
    for agent in agents:
        proposal = next(proposal for proposal in idea_drafts if proposal.agent_name == agent.name)
        reviews = [review for review in idea_reviews if review.target_proposal_id == proposal.proposal_id]
        idea_revisions.append(agent.revise_idea(brief, proposal, reviews))

    hypotheses = _synthesize_hypotheses(
        trading_idea=trading_idea,
        revisions=idea_revisions,
        client=client,
        model=model,
        target_count=target_count,
        retries=json_retries,
    )

    artifacts = {
        "idea_brief.json": brief,
        "idea_drafts.json": idea_drafts,
        "idea_reviews.json": idea_reviews,
        "idea_revisions.json": idea_revisions,
        "hypotheses.json": hypotheses,
    }
    return hypotheses, artifacts


def run_formula_debate(
    hypotheses: list[ResearchHypothesisSpec],
    available_terminals: list[str],
    client: OpenAI,
    model: str,
    config: Config | None = None,
) -> tuple[SeedFormulaPack, dict[str, Any]]:
    """Run Stage 2 formula debate and return the seed formula pack."""
    json_retries = _config_value(config, "debate_json_retries", 2)
    target_count = _config_value(config, "seed_formula_target", 8)
    agents = create_agents(client, model, json_retries=json_retries)
    formula_briefs = _build_formula_briefs(hypotheses, available_terminals)
    brief_map = {brief.hypothesis_id: brief for brief in formula_briefs}

    formula_drafts: list[FormulaProposal] = []
    for agent in agents:
        formula_drafts.extend(agent.draft_formulas(formula_briefs))

    formula_reviews: list[FormulaReview] = []
    for agent in agents:
        targets = [formula for formula in formula_drafts if formula.agent_name != agent.name]
        formula_reviews.extend(agent.review_formulas(targets))

    formula_revisions: list[FormulaRevision] = []
    for agent in agents:
        own_formulas = [formula for formula in formula_drafts if formula.agent_name == agent.name]
        own_reviews = [
            review for review in formula_reviews
            if any(formula.formula_id == review.target_formula_id for formula in own_formulas)
        ]
        formula_revisions.extend(agent.revise_formulas(own_formulas, own_reviews, brief_map))

    revised_formulas = [revision.revised_formula for revision in formula_revisions]
    _annotate_formula_parse_status(revised_formulas, available_terminals)
    seed_pack = _select_seed_formulas(
        hypotheses=hypotheses,
        formulas=revised_formulas,
        client=client,
        model=model,
        target_count=target_count,
        retries=json_retries,
    )

    artifacts = {
        "formula_briefs.json": formula_briefs,
        "formula_drafts.json": formula_drafts,
        "formula_reviews.json": formula_reviews,
        "formula_revisions.json": formula_revisions,
        "seed_formula_pack.json": seed_pack,
        "seed_formulas.json": seed_pack.selected_formulas,
    }
    return seed_pack, artifacts


def run_debate(
    trading_idea: str,
    client: OpenAI,
    model: str,
    num_rounds: int = 2,
    available_terminals: list[str] | None = None,
    config: Config | None = None,
) -> list[dict]:
    """Backward-compatible wrapper that returns a flat seed list."""
    del num_rounds
    if available_terminals is None:
        try:
            from alpha_gpt.gp_search.primitives import DEFAULT_TERMINALS

            terminals = list(DEFAULT_TERMINALS)
        except ModuleNotFoundError:
            terminals = []
    else:
        terminals = available_terminals
    hypotheses, _ = run_idea_debate(trading_idea, terminals, client, model, config=config)
    seed_pack, _ = run_formula_debate(hypotheses, terminals, client, model, config=config)
    return [
        {
            "expression": formula.expression,
            "description": formula.plain_language_mapping or formula.rationale,
        }
        for formula in seed_pack.selected_formulas
    ]
