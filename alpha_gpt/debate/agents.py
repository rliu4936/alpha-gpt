"""Debate agents: three LLM agents with distinct personas for alpha generation."""

import json
import logging

from openai import OpenAI

from alpha_gpt.debate.prompts import (
    MOMENTUM_SYSTEM, MEAN_REVERSION_SYSTEM, FUNDAMENTAL_SYSTEM,
    ROUND1_USER, ROUND2_USER,
)

logger = logging.getLogger(__name__)


class DebateAgent:
    """A single debate agent with a specific persona."""

    def __init__(self, name: str, system_prompt: str, client: OpenAI, model: str):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.model = model

    def generate_alphas(
        self,
        trading_idea: str,
        round_num: int,
        prior_proposals: str = "",
    ) -> list[dict]:
        """Generate alpha proposals for a given trading idea.

        Args:
            trading_idea: The trading idea to generate alphas for.
            round_num: 1 for initial proposals, 2 for revision round.
            prior_proposals: JSON string of prior proposals (for round 2).

        Returns:
            List of dicts with 'expression', 'description', 'rationale'.
        """
        if round_num == 1:
            user_msg = ROUND1_USER.format(trading_idea=trading_idea)
        else:
            user_msg = ROUND2_USER.format(
                trading_idea=trading_idea,
                prior_proposals=prior_proposals,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            content = response.choices[0].message.content.strip()
            alphas = _parse_json_response(content)
            logger.info(f"[{self.name}] Round {round_num}: generated {len(alphas)} alphas")
            return alphas

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
            return []


def create_agents(client: OpenAI, model: str) -> list[DebateAgent]:
    """Create the three debate agents with distinct personas."""
    return [
        DebateAgent("Momentum", MOMENTUM_SYSTEM, client, model),
        DebateAgent("MeanReversion", MEAN_REVERSION_SYSTEM, client, model),
        DebateAgent("Fundamental", FUNDAMENTAL_SYSTEM, client, model),
    ]


def _parse_json_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Strip markdown code fences
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    content = content.strip()

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON: {content[:200]}")
        return []
