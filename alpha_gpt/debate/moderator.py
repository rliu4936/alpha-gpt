"""Debate moderator: orchestrates multi-round debate and selects best seed alphas."""

import json
import logging

from openai import OpenAI

from alpha_gpt.debate.agents import DebateAgent, create_agents, _parse_json_response
from alpha_gpt.debate.prompts import MODERATOR_SYSTEM, MODERATOR_USER, OPERATOR_CATALOG

logger = logging.getLogger(__name__)


def run_debate(
    trading_idea: str,
    client: OpenAI,
    model: str,
    num_rounds: int = 2,
) -> list[dict]:
    """Run a multi-round debate to generate seed alpha expressions.

    Args:
        trading_idea: Natural language trading idea.
        client: OpenAI client (pointed at OpenRouter).
        model: Model identifier.
        num_rounds: Number of debate rounds (default 2).

    Returns:
        List of validated seed alpha dicts with 'expression' and 'description'.
    """
    agents = create_agents(client, model)
    all_proposals = []

    # Round 1: Independent proposals
    logger.info(f"=== Debate Round 1: Independent proposals ===")
    print(f"\n--- Debate Round 1 ---")
    for agent in agents:
        proposals = agent.generate_alphas(trading_idea, round_num=1)
        for p in proposals:
            p["source"] = agent.name
            p["round"] = 1
        all_proposals.extend(proposals)
        print(f"  [{agent.name}] proposed {len(proposals)} alphas")

    # Round 2+: Each agent sees all prior proposals and can revise
    for round_num in range(2, num_rounds + 1):
        logger.info(f"=== Debate Round {round_num}: Revision ===")
        print(f"\n--- Debate Round {round_num} ---")
        prior_json = json.dumps(
            [{"source": p.get("source", "?"), "expression": p.get("expression", ""),
              "description": p.get("description", "")}
             for p in all_proposals],
            indent=2,
        )
        for agent in agents:
            proposals = agent.generate_alphas(
                trading_idea, round_num=round_num, prior_proposals=prior_json
            )
            for p in proposals:
                p["source"] = agent.name
                p["round"] = round_num
            all_proposals.extend(proposals)
            print(f"  [{agent.name}] proposed {len(proposals)} alphas")

    print(f"\nTotal proposals: {len(all_proposals)}")

    # Moderation: select top seeds
    seeds = _moderate(all_proposals, client, model)
    print(f"Moderator selected {len(seeds)} seed alphas")

    return seeds


def _moderate(all_proposals: list[dict], client: OpenAI, model: str) -> list[dict]:
    """Use LLM to deduplicate, validate, and select top seed alphas."""
    proposals_json = json.dumps(
        [{"source": p.get("source", "?"), "expression": p.get("expression", ""),
          "description": p.get("description", ""), "rationale": p.get("rationale", "")}
         for p in all_proposals],
        indent=2,
    )

    user_msg = MODERATOR_USER.format(
        all_proposals=proposals_json,
        operator_catalog=OPERATOR_CATALOG,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MODERATOR_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        seeds = _parse_json_response(content)
        logger.info(f"Moderator selected {len(seeds)} seeds")
        return seeds

    except Exception as e:
        logger.error(f"Moderator error: {e}")
        # Fallback: return all unique expressions from round 1
        seen = set()
        fallback = []
        for p in all_proposals:
            expr = p.get("expression", "")
            if expr and expr not in seen:
                seen.add(expr)
                fallback.append({"expression": expr, "description": p.get("description", "")})
        return fallback[:8]
