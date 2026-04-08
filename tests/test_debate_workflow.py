import json
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch
import unittest

import pandas as pd

from alpha_gpt.config import Config
from alpha_gpt.data.loader import DataSplit
from alpha_gpt.debate.models import (
    FormulaProposal,
    FormulaRevision,
    FormulaReview,
    IdeaProposal,
    IdeaRevision,
    IdeaReview,
    ResearchHypothesisSpec,
    SeedFormulaPack,
    make_id,
)
from alpha_gpt.debate.moderator import run_debate, run_formula_debate, run_idea_debate


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)

    def create(self, **kwargs):
        return _FakeResponse(self.contents.pop(0))


class _FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=_FakeCompletions(contents))


class _StubAgent:
    def __init__(self, name: str):
        self.name = name

    def draft_idea(self, brief):
        return IdeaProposal(
            proposal_id=make_id("idea-draft", self.name),
            agent_name=self.name,
            title=f"{self.name} hypothesis",
            mechanism=f"{self.name} mechanism",
            signal_type="cross_sectional_alpha",
            payoff_definition="Predicts relative returns.",
            directionality="directional",
            direction_separation_plan="Not needed.",
            data_definition="Use price and volume proxies.",
            candidate_proxies=["close", "volume"],
            subfactor_design=[f"{self.name} subfactor"],
            filter_policy="No extra filter.",
            normalization_policy="cs_rank",
            neutralization_policy="none",
            implementability="directly_supported",
            open_risks=["crowding"],
            stage2_constraints=["Keep formulas simple."],
            summary=f"{self.name} summary",
        )

    def review_ideas(self, brief, proposals):
        return [
            IdeaReview(
                review_id=make_id("idea-review", self.name, proposal.proposal_id),
                reviewer_agent_name=self.name,
                target_proposal_id=proposal.proposal_id,
                mechanism_quality=4,
                signal_type_clarity=4,
                payoff_clarity=4,
                directionality_clarity=4,
                subfactor_quality=4,
                filter_logic=4,
                normalization_soundness=4,
                implementability=4,
                decision="accept_with_revision",
                comments=[f"Review from {self.name}"],
            )
            for proposal in proposals
        ]

    def revise_idea(self, brief, proposal, reviews):
        return IdeaRevision(
            revision_id=make_id("idea-revision", self.name),
            agent_name=self.name,
            base_proposal_id=proposal.proposal_id,
            accepted_feedback=[review.comments[0] for review in reviews],
            revision_summary=f"{self.name} revised",
            revised_proposal=proposal,
        )

    def draft_formulas(self, briefs):
        proposals = []
        for idx, brief in enumerate(briefs, start=1):
            if self.name == "MeanReversion" and idx == 1:
                expression = "not_a_formula(close)"
            elif self.name == "Fundamental" and idx == 1:
                expression = "cs_rank(volume)"
            else:
                expression = "cs_rank(close)"
            role = "filter" if self.name == "Fundamental" and idx == 1 else "main_alpha"
            proposals.append(
                FormulaProposal(
                    formula_id=make_id("formula", self.name, brief.hypothesis_id, str(idx)),
                    hypothesis_id=brief.hypothesis_id,
                    agent_name=self.name,
                    formula_role=role,
                    expression=expression,
                    plain_language_mapping=f"{self.name} formula for {brief.hypothesis_id}",
                    terminals_used=["close"],
                    operators_used=["cs_rank"],
                    expected_signal_direction="positive",
                    rationale="Stub rationale",
                )
            )
        return proposals

    def review_formulas(self, proposals):
        return [
            FormulaReview(
                review_id=make_id("formula-review", self.name, proposal.formula_id),
                reviewer_agent_name=self.name,
                target_formula_id=proposal.formula_id,
                faithfulness=4,
                implementability=4,
                robustness=4,
                novelty=4,
                simplicity=4,
                decision="accept_with_revision",
                comments=[f"Formula review from {self.name}"],
            )
            for proposal in proposals
        ]

    def revise_formulas(self, proposals, reviews, briefs_by_id):
        return [
            FormulaRevision(
                revision_id=make_id("formula-revision", self.name, proposal.formula_id),
                agent_name=self.name,
                base_formula_id=proposal.formula_id,
                accepted_feedback=[review.comments[0] for review in reviews if review.target_formula_id == proposal.formula_id],
                revision_summary=f"{self.name} revised formula",
                revised_formula=proposal,
            )
            for proposal in proposals
        ]


class _DummyLogbook:
    def select(self, key):
        return {"gen": [0], "max": [0.1], "avg": [0.1]}[key]


class DebateWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.idea_hypothesis_target = 2
        self.config.seed_formula_target = 5
        self.stub_agents = [
            _StubAgent("Momentum"),
            _StubAgent("MeanReversion"),
            _StubAgent("Fundamental"),
        ]

    def test_run_idea_debate_produces_expected_artifact_counts(self):
        client = _FakeClient([
            """
            [
              {
                "title": "Hypothesis Alpha",
                "source_agents": ["Momentum", "Fundamental"],
                "mechanism": "Mechanism A",
                "signal_type": "cross_sectional_alpha",
                "payoff_definition": "Relative return",
                "directionality": "directional",
                "direction_separation_plan": "Not needed",
                "data_definition": "close and volume",
                "candidate_proxies": ["close"],
                "subfactor_design": ["sub_a"],
                "filter_policy": "none",
                "normalization_policy": "cs_rank",
                "neutralization_policy": "none",
                "implementability": "directly_supported",
                "open_risks": ["crowding"],
                "stage2_constraints": ["simple"],
                "summary": "A"
              },
              {
                "title": "Hypothesis Beta",
                "source_agents": ["MeanReversion"],
                "mechanism": "Mechanism B",
                "signal_type": "filter",
                "payoff_definition": "Conditioning",
                "directionality": "non_directional",
                "direction_separation_plan": "Use as filter",
                "data_definition": "returns dispersion",
                "candidate_proxies": ["returns"],
                "subfactor_design": ["sub_b"],
                "filter_policy": "regime gate",
                "normalization_policy": "zscore",
                "neutralization_policy": "none",
                "implementability": "directly_supported",
                "open_risks": ["lag"],
                "stage2_constraints": ["allow filter role"],
                "summary": "B"
              }
            ]
            """
        ])

        with patch("alpha_gpt.debate.moderator.create_agents", return_value=self.stub_agents):
            hypotheses, artifacts = run_idea_debate(
                trading_idea="test idea",
                available_terminals=["close", "volume", "returns"],
                client=client,
                model="fake-model",
                config=self.config,
            )

        self.assertEqual(len(artifacts["idea_drafts.json"]), 3)
        self.assertEqual(len(artifacts["idea_reviews.json"]), 6)
        self.assertEqual(len(artifacts["idea_revisions.json"]), 3)
        self.assertEqual(len(hypotheses), 2)

    def test_run_formula_debate_keeps_parseable_filter_formulas(self):
        hypotheses = [
            ResearchHypothesisSpec(
                hypothesis_id="hypothesis-alpha",
                title="Hypothesis Alpha",
                summary="Summary A",
                stage2_constraints=["simple"],
            ),
            ResearchHypothesisSpec(
                hypothesis_id="hypothesis-beta",
                title="Hypothesis Beta",
                summary="Summary B",
                stage2_constraints=["allow filter role"],
            ),
        ]
        selected_ids = [
            make_id("formula", "Momentum", "hypothesis-alpha", "1"),
            make_id("formula", "Momentum", "hypothesis-beta", "2"),
            make_id("formula", "MeanReversion", "hypothesis-beta", "2"),
            make_id("formula", "Fundamental", "hypothesis-alpha", "1"),
            make_id("formula", "Fundamental", "hypothesis-beta", "2"),
        ]
        selected_filter_id = make_id("formula", "Fundamental", "hypothesis-alpha", "1")
        client = _FakeClient([
            f"""
            {{
              "selected_formula_ids": {json.dumps(selected_ids)},
              "selection_rationale": ["diverse roles", "cross-hypothesis coverage"]
            }}
            """
        ])

        with patch("alpha_gpt.debate.moderator.create_agents", return_value=self.stub_agents):
            seed_pack, artifacts = run_formula_debate(
                hypotheses=hypotheses,
                available_terminals=["close", "volume", "returns"],
                client=client,
                model="fake-model",
                config=self.config,
            )

        self.assertTrue(seed_pack.selected_formulas)
        self.assertEqual(len(seed_pack.selected_formulas), 5)
        self.assertTrue(all(formula.parseable for formula in seed_pack.selected_formulas))
        self.assertTrue(any(formula.formula_role == "filter" for formula in seed_pack.selected_formulas))
        self.assertIn(selected_filter_id, [formula.formula_id for formula in seed_pack.selected_formulas])
        self.assertTrue(any(not revision.revised_formula.parseable for revision in artifacts["formula_revisions.json"]))

    def test_run_debate_wrapper_returns_flat_legacy_shape(self):
        seed_pack = SeedFormulaPack(
            pack_id="pack",
            selected_formulas=[
                FormulaProposal(
                    formula_id="formula-1",
                    hypothesis_id="hyp-1",
                    agent_name="Momentum",
                    expression="cs_rank(close)",
                    plain_language_mapping="legacy-compatible description",
                )
            ],
        )
        with patch("alpha_gpt.debate.moderator.run_idea_debate", return_value=([ResearchHypothesisSpec(hypothesis_id="hyp-1")], {})):
            with patch("alpha_gpt.debate.moderator.run_formula_debate", return_value=(seed_pack, {})):
                result = run_debate("idea", client=MagicMock(), model="fake-model")

        self.assertEqual(result, [{"expression": "cs_rank(close)", "description": "legacy-compatible description"}])

    def test_main_pipeline_uses_two_stage_debate_and_flattens_seed_expressions(self):
        try:
            from alpha_gpt.main import run_pipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"alpha_gpt.main could not be imported in this environment: {exc}")

        idx = pd.to_datetime(["2020-01-01"])
        df = pd.DataFrame([[1.0]], index=idx, columns=[10001])
        split = DataSplit(panels={"close": df, "returns": df}, forward_returns=df)
        hypotheses = [ResearchHypothesisSpec(hypothesis_id="hyp-1", title="H1", summary="S1")]
        seed_pack = SeedFormulaPack(
            pack_id="pack",
            selected_formulas=[
                FormulaProposal(
                    formula_id="formula-1",
                    hypothesis_id="hyp-1",
                    agent_name="Momentum",
                    expression="cs_rank(close)",
                    plain_language_mapping="alpha one",
                    parseable=True,
                ),
                FormulaProposal(
                    formula_id="formula-2",
                    hypothesis_id="hyp-1",
                    agent_name="Fundamental",
                    expression="cs_rank(volume)",
                    plain_language_mapping="alpha two",
                    parseable=True,
                ),
            ],
        )
        bt = SimpleNamespace(
            sharpe=1.0,
            annual_return=0.1,
            max_drawdown=-0.1,
            quantile_returns=pd.DataFrame({"mean_daily_return": [0.1], "count": [1]}, index=["Q1"]),
            cumulative_returns=pd.Series([1.0], index=idx),
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch("alpha_gpt.main.load_panels", return_value={"close": df, "returns": df, "forward_returns": df}):
                with patch("alpha_gpt.main.split_data", return_value=(split, split, split)):
                    with patch("alpha_gpt.main.OpenAI", return_value=MagicMock()):
                        with patch("alpha_gpt.main.run_idea_debate", return_value=(hypotheses, {"idea_brief.json": {"ok": True}})):
                            with patch("alpha_gpt.main.run_formula_debate", return_value=(seed_pack, {"seed_formula_pack.json": {"ok": True}})):
                                with patch("alpha_gpt.main.inject_seeds", return_value=["tree"]) as mock_inject:
                                    with patch("alpha_gpt.main.run_gp", return_value=([{"expression": "cs_rank(close)", "tree": "tree", "fitness": 0.1}], _DummyLogbook())):
                                        with patch("alpha_gpt.main._eval_expr", return_value=df):
                                            with patch("alpha_gpt.main.compute_ic", return_value=pd.Series([0.1], index=idx)):
                                                with patch("alpha_gpt.main.compute_icir", return_value=0.2):
                                                    with patch("alpha_gpt.main.compute_turnover", return_value=0.3):
                                                        with patch("alpha_gpt.main.backtest_alpha", return_value=bt):
                                                            with patch("alpha_gpt.main.explain_alpha", return_value="ok"):
                                                                with patch("alpha_gpt.main.plot_gp_evolution"):
                                                                    with patch("alpha_gpt.main.plot_equity_curves"):
                                                                        with patch("alpha_gpt.main.os.makedirs"):
                                                                            with patch("builtins.open", mock_open()):
                                                                                run_pipeline("idea", config=self.config)

        mock_inject.assert_called_once()
        self.assertEqual(mock_inject.call_args.args[0], ["cs_rank(close)", "cs_rank(volume)"])
