"""Stage-aware debate agents for the two-stage framework."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from openai import OpenAI

from alpha_gpt.debate.models import (
    FormulaDebateBrief,
    FormulaProposal,
    FormulaRevision,
    FormulaReview,
    IdeaDebateBrief,
    IdeaProposal,
    IdeaRevision,
    IdeaReview,
    VALID_FORMULA_ROLES,
    VALID_REVIEW_DECISIONS,
    make_id,
    to_jsonable,
)
from alpha_gpt.debate.prompts import (
    FORMULA_DRAFT_USER,
    FORMULA_REVIEW_USER,
    FORMULA_REVISION_USER,
    FUNDAMENTAL_SYSTEM,
    IDEA_DRAFT_USER,
    IDEA_REVIEW_USER,
    IDEA_REVISION_USER,
    JSON_REPAIR_USER,
    MEAN_REVERSION_SYSTEM,
    MOMENTUM_SYSTEM,
)

logger = logging.getLogger(__name__)


def _extract_json_payload(content: str) -> Any | None:
    """Parse a JSON object or array from a model response."""
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]

    content = content.strip()
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = content.find(opener)
        end = content.rfind(closer)
        if start >= 0 and end > start:
            snippet = content[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    return None


def _coerce_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_coerce_str(item) for item in value if _coerce_str(item)]
    text = _coerce_str(value)
    return [text] if text else []


def _coerce_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(5, score))


class DebateAgent:
    """A full research agent that participates in both debate stages."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        client: OpenAI,
        model: str,
        json_retries: int = 2,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.json_retries = max(1, json_retries)

    def _call_json(
        self,
        user_prompt: str,
        empty_factory: Callable[[], Any],
        temperature: float = 0.5,
    ) -> Any:
        """Call the model and parse JSON with one repair attempt."""
        invalid_response = ""
        for attempt in range(self.json_retries):
            if attempt == 0:
                prompt = user_prompt
            else:
                prompt = JSON_REPAIR_USER.format(
                    original_prompt=user_prompt,
                    invalid_response=invalid_response,
                )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=temperature if attempt == 0 else 0.0,
                )
                invalid_response = response.choices[0].message.content.strip()
                payload = _extract_json_payload(invalid_response)
                if payload is not None:
                    return payload
            except Exception as exc:
                logger.error("[%s] JSON call failed on attempt %s: %s", self.name, attempt + 1, exc)
                break

        return empty_factory()

    def draft_idea(self, brief: IdeaDebateBrief) -> IdeaProposal:
        user_msg = IDEA_DRAFT_USER.format(
            trading_idea=brief.trading_idea,
            available_terminals=", ".join(brief.available_terminals) or "none",
            constraints=json.dumps(brief.constraints, indent=2),
            data_notes=json.dumps(brief.data_notes, indent=2),
        )
        payload = self._call_json(user_msg, empty_factory=dict, temperature=0.7)
        proposal_id = make_id("idea-draft", self.name)
        return self._idea_proposal_from_payload(payload, proposal_id)

    def review_ideas(self, brief: IdeaDebateBrief, proposals: list[IdeaProposal]) -> list[IdeaReview]:
        if not proposals:
            return []

        user_msg = IDEA_REVIEW_USER.format(
            trading_idea=brief.trading_idea,
            proposals_json=json.dumps(to_jsonable(proposals), indent=2, ensure_ascii=False),
        )
        payload = self._call_json(user_msg, empty_factory=list, temperature=0.3)
        reviews = payload if isinstance(payload, list) else []
        if len(reviews) < len(proposals):
            reviews = list(reviews) + ([{}] * (len(proposals) - len(reviews)))
        return [
            self._idea_review_from_payload(item, proposals, idx)
            for idx, item in enumerate(reviews[: len(proposals)])
        ]

    def revise_idea(
        self,
        brief: IdeaDebateBrief,
        proposal: IdeaProposal,
        reviews: list[IdeaReview],
    ) -> IdeaRevision:
        user_msg = IDEA_REVISION_USER.format(
            trading_idea=brief.trading_idea,
            proposal_json=json.dumps(to_jsonable(proposal), indent=2, ensure_ascii=False),
            reviews_json=json.dumps(to_jsonable(reviews), indent=2, ensure_ascii=False),
        )
        payload = self._call_json(user_msg, empty_factory=dict, temperature=0.4)
        revised_payload = payload.get("revised_proposal", {}) if isinstance(payload, dict) else {}
        revised_proposal = self._idea_proposal_from_payload(revised_payload, proposal_id=proposal.proposal_id)
        return IdeaRevision(
            revision_id=make_id("idea-revision", self.name),
            agent_name=self.name,
            base_proposal_id=proposal.proposal_id,
            accepted_feedback=_coerce_list_of_str(payload.get("accepted_feedback") if isinstance(payload, dict) else []),
            rejected_feedback=_coerce_list_of_str(payload.get("rejected_feedback") if isinstance(payload, dict) else []),
            revision_summary=_coerce_str(payload.get("revision_summary") if isinstance(payload, dict) else ""),
            revised_proposal=revised_proposal,
        )

    def draft_formulas(self, briefs: list[FormulaDebateBrief]) -> list[FormulaProposal]:
        if not briefs:
            return []

        user_msg = FORMULA_DRAFT_USER.format(
            available_terminals=", ".join(sorted({t for brief in briefs for t in brief.available_terminals})) or "none",
            hypotheses_json=json.dumps(to_jsonable(briefs), indent=2, ensure_ascii=False),
        )
        payload = self._call_json(user_msg, empty_factory=list, temperature=0.7)
        proposals = payload if isinstance(payload, list) else []
        brief_map = {brief.hypothesis_id: brief for brief in briefs}
        result = []
        for idx, item in enumerate(proposals):
            formula = self._formula_proposal_from_payload(item, brief_map, idx)
            if formula.expression:
                result.append(formula)
        return result

    def review_formulas(self, proposals: list[FormulaProposal]) -> list[FormulaReview]:
        if not proposals:
            return []

        user_msg = FORMULA_REVIEW_USER.format(
            proposals_json=json.dumps(to_jsonable(proposals), indent=2, ensure_ascii=False),
        )
        payload = self._call_json(user_msg, empty_factory=list, temperature=0.3)
        reviews = payload if isinstance(payload, list) else []
        if len(reviews) < len(proposals):
            reviews = list(reviews) + ([{}] * (len(proposals) - len(reviews)))
        return [
            self._formula_review_from_payload(item, proposals, idx)
            for idx, item in enumerate(reviews[: len(proposals)])
        ]

    def revise_formulas(
        self,
        proposals: list[FormulaProposal],
        reviews: list[FormulaReview],
        briefs_by_id: dict[str, FormulaDebateBrief],
    ) -> list[FormulaRevision]:
        if not proposals:
            return []

        user_msg = FORMULA_REVISION_USER.format(
            proposals_json=json.dumps(to_jsonable(proposals), indent=2, ensure_ascii=False),
            reviews_json=json.dumps(to_jsonable(reviews), indent=2, ensure_ascii=False),
        )
        payload = self._call_json(user_msg, empty_factory=list, temperature=0.4)
        items = payload if isinstance(payload, list) else []
        if len(items) < len(proposals):
            items = list(items) + ([{}] * (len(proposals) - len(items)))
        revisions = []
        for idx, item in enumerate(items[: len(proposals)]):
            base_formula = proposals[idx]
            base_formula_id = _coerce_str(item.get("base_formula_id")) if isinstance(item, dict) else ""
            if not base_formula_id:
                base_formula_id = base_formula.formula_id
            revised_payload = item.get("revised_formula", {}) if isinstance(item, dict) else {}
            revised_formula = self._formula_proposal_from_payload(
                revised_payload,
                briefs_by_id,
                idx,
                fallback_formula=base_formula,
            )
            revisions.append(
                FormulaRevision(
                    revision_id=make_id("formula-revision", self.name, base_formula_id),
                    agent_name=self.name,
                    base_formula_id=base_formula_id,
                    accepted_feedback=_coerce_list_of_str(item.get("accepted_feedback") if isinstance(item, dict) else []),
                    rejected_feedback=_coerce_list_of_str(item.get("rejected_feedback") if isinstance(item, dict) else []),
                    revision_summary=_coerce_str(item.get("revision_summary") if isinstance(item, dict) else ""),
                    revised_formula=revised_formula,
                )
            )

        if revisions:
            return revisions

        return [
            FormulaRevision(
                revision_id=make_id("formula-revision", self.name, proposal.formula_id),
                agent_name=self.name,
                base_formula_id=proposal.formula_id,
                revision_summary="No revision returned; kept original formula.",
                revised_formula=proposal,
            )
            for proposal in proposals
        ]

    def _idea_proposal_from_payload(self, payload: Any, proposal_id: str) -> IdeaProposal:
        payload = payload if isinstance(payload, dict) else {}
        return IdeaProposal(
            proposal_id=proposal_id,
            agent_name=self.name,
            title=_coerce_str(payload.get("title")) or f"{self.name} hypothesis draft",
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

    def _idea_review_from_payload(
        self,
        payload: Any,
        proposals: list[IdeaProposal],
        idx: int,
    ) -> IdeaReview:
        payload = payload if isinstance(payload, dict) else {}
        target = proposals[min(idx, len(proposals) - 1)]
        target_id = _coerce_str(payload.get("target_proposal_id")) or target.proposal_id
        decision = _coerce_str(payload.get("decision")) or "accept_with_revision"
        if decision not in VALID_REVIEW_DECISIONS:
            decision = "accept_with_revision"
        return IdeaReview(
            review_id=make_id("idea-review", self.name, target_id),
            reviewer_agent_name=self.name,
            target_proposal_id=target_id,
            mechanism_quality=_coerce_score(payload.get("mechanism_quality")),
            signal_type_clarity=_coerce_score(payload.get("signal_type_clarity")),
            payoff_clarity=_coerce_score(payload.get("payoff_clarity")),
            directionality_clarity=_coerce_score(payload.get("directionality_clarity")),
            subfactor_quality=_coerce_score(payload.get("subfactor_quality")),
            filter_logic=_coerce_score(payload.get("filter_logic")),
            normalization_soundness=_coerce_score(payload.get("normalization_soundness")),
            implementability=_coerce_score(payload.get("implementability")),
            decision=decision,
            comments=_coerce_list_of_str(payload.get("comments")),
        )

    def _formula_proposal_from_payload(
        self,
        payload: Any,
        briefs_by_id: dict[str, FormulaDebateBrief],
        idx: int,
        fallback_formula: FormulaProposal | None = None,
    ) -> FormulaProposal:
        payload = payload if isinstance(payload, dict) else {}
        hypothesis_ids = list(briefs_by_id.keys())
        fallback_hypothesis_id = fallback_formula.hypothesis_id if fallback_formula else (
            hypothesis_ids[min(idx, len(hypothesis_ids) - 1)] if hypothesis_ids else ""
        )
        hypothesis_id = _coerce_str(payload.get("hypothesis_id")) or fallback_hypothesis_id
        formula_role = _coerce_str(payload.get("formula_role")) or (
            fallback_formula.formula_role if fallback_formula else "main_alpha"
        )
        if formula_role not in VALID_FORMULA_ROLES:
            formula_role = "main_alpha"
        formula_id = (
            fallback_formula.formula_id
            if fallback_formula is not None
            else make_id("formula", self.name, hypothesis_id, str(idx + 1))
        )
        return FormulaProposal(
            formula_id=formula_id,
            hypothesis_id=hypothesis_id,
            agent_name=self.name,
            formula_role=formula_role,
            expression=_coerce_str(payload.get("expression")) or (fallback_formula.expression if fallback_formula else ""),
            plain_language_mapping=_coerce_str(payload.get("plain_language_mapping")) or (
                fallback_formula.plain_language_mapping if fallback_formula else ""
            ),
            terminals_used=_coerce_list_of_str(payload.get("terminals_used")) or (
                list(fallback_formula.terminals_used) if fallback_formula else []
            ),
            operators_used=_coerce_list_of_str(payload.get("operators_used")) or (
                list(fallback_formula.operators_used) if fallback_formula else []
            ),
            expected_signal_direction=_coerce_str(payload.get("expected_signal_direction")) or (
                fallback_formula.expected_signal_direction if fallback_formula else ""
            ),
            embedded_filter_logic=_coerce_str(payload.get("embedded_filter_logic")) or (
                fallback_formula.embedded_filter_logic if fallback_formula else ""
            ),
            normalization_in_formula=_coerce_str(payload.get("normalization_in_formula")) or (
                fallback_formula.normalization_in_formula if fallback_formula else ""
            ),
            neutralization_in_formula_or_postprocess=_coerce_str(
                payload.get("neutralization_in_formula_or_postprocess")
            ) or (
                fallback_formula.neutralization_in_formula_or_postprocess if fallback_formula else ""
            ),
            rationale=_coerce_str(payload.get("rationale")) or (
                fallback_formula.rationale if fallback_formula else ""
            ),
        )

    def _formula_review_from_payload(
        self,
        payload: Any,
        proposals: list[FormulaProposal],
        idx: int,
    ) -> FormulaReview:
        payload = payload if isinstance(payload, dict) else {}
        target = proposals[min(idx, len(proposals) - 1)]
        target_id = _coerce_str(payload.get("target_formula_id")) or target.formula_id
        decision = _coerce_str(payload.get("decision")) or "accept_with_revision"
        if decision not in VALID_REVIEW_DECISIONS:
            decision = "accept_with_revision"
        return FormulaReview(
            review_id=make_id("formula-review", self.name, target_id),
            reviewer_agent_name=self.name,
            target_formula_id=target_id,
            faithfulness=_coerce_score(payload.get("faithfulness")),
            implementability=_coerce_score(payload.get("implementability")),
            robustness=_coerce_score(payload.get("robustness")),
            novelty=_coerce_score(payload.get("novelty")),
            simplicity=_coerce_score(payload.get("simplicity")),
            decision=decision,
            comments=_coerce_list_of_str(payload.get("comments")),
        )


def create_agents(client: OpenAI, model: str, json_retries: int = 2) -> list[DebateAgent]:
    """Create the three style-based full research agents."""
    return [
        DebateAgent("Momentum", MOMENTUM_SYSTEM, client, model, json_retries=json_retries),
        DebateAgent("MeanReversion", MEAN_REVERSION_SYSTEM, client, model, json_retries=json_retries),
        DebateAgent("Fundamental", FUNDAMENTAL_SYSTEM, client, model, json_retries=json_retries),
    ]
