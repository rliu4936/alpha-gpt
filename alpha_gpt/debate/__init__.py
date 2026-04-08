"""Debate framework exports."""

__all__ = ["run_debate", "run_idea_debate", "run_formula_debate"]


def __getattr__(name):
    if name in __all__:
        from alpha_gpt.debate.moderator import run_debate, run_formula_debate, run_idea_debate

        return {
            "run_debate": run_debate,
            "run_idea_debate": run_idea_debate,
            "run_formula_debate": run_formula_debate,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
