"""Configuration for Alpha-GPT 3.0 pipeline."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Data
    data_dir: str = "data/panels"

    # LLM via OpenRouter/OpenAI
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-v3.2"

    # GP search
    gp_population: int = 100
    gp_generations: int = 20
    gp_crossover: float = 0.7
    gp_mutation: float = 0.2
    gp_tournament_size: int = 3
    gp_max_depth: int = 6

    # Debate
    debate_rounds: int = 2
    idea_hypothesis_target: int = 3
    seed_formula_target: int = 8
    debate_json_retries: int = 2
    save_debate_artifacts: bool = True

    # Evaluation
    top_k: int = 5
    train_end: str = "2017-12-31"
    val_end: str = "2020-12-31"

    @property
    def openrouter_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")
        return key
