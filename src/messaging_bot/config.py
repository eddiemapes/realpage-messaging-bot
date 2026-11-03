"""Settings shared across the bot: environment variables, model names and file paths."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# The .env file lives in the repository root, one level above this uv project.
# find_dotenv walks up from the current working directory until it finds it.
load_dotenv(find_dotenv(usecwd=True))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SAMPLE_DATA_PATH = DATA_DIR / "sample.jsonl"
HOLDOUT_DATA_PATH = DATA_DIR / "holdout.jsonl"
DEFAULT_DATA_PATHS = [SAMPLE_DATA_PATH, HOLDOUT_DATA_PATH]

DECIDER_MODEL = os.environ.get("DECIDER_MODEL", "gpt-4.1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4.1")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3


def load_prompt(file_name: str) -> str:
    """Read a prompt template from the prompts folder."""
    return (PROMPTS_DIR / file_name).read_text(encoding="utf-8")
