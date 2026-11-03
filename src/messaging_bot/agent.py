"""The decision agent: retrieved examples + target record -> structured BotDecision."""

import json
import time
from functools import lru_cache

from langfuse import observe
from langfuse.openai import OpenAI

from messaging_bot.config import DECIDER_MODEL, load_prompt
from messaging_bot.models import BotDecision, BotRun, RetrievedExample, TestCase
from messaging_bot.records import record_for_prompt
from messaging_bot.retrieval import ExampleIndex


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """One traced OpenAI client per process (reads OPENAI_API_KEY from the environment)."""
    return OpenAI()


def format_examples(examples: list[TestCase]) -> str:
    """Render worked examples as labelled input/expected JSON blocks."""
    if not examples:
        return "No worked examples are available. Rely on the general rules."

    example_blocks = []
    for example_number, example in enumerate(examples, start=1):
        example_input = json.dumps(record_for_prompt(example), indent=2, ensure_ascii=False)
        expected_outcome = json.dumps(example.expected.model_dump(mode="json"), indent=2, ensure_ascii=False)
        example_blocks.append(
            f"### Example {example_number}\n"
            f"Input:\n```json\n{example_input}\n```\n"
            f"Expected outcome:\n```json\n{expected_outcome}\n```"
        )
    return "\n\n".join(example_blocks)


def build_messages(record: TestCase, examples: list[TestCase]) -> list[dict[str, str]]:
    target_input = json.dumps(record_for_prompt(record), indent=2, ensure_ascii=False)
    user_message = (
        "## Worked examples\n\n"
        f"{format_examples(examples)}\n\n"
        "## Record to decide on\n\n"
        f"```json\n{target_input}\n```\n\n"
        "Decide the outcome for this record."
    )
    return [
        {"role": "system", "content": load_prompt("decider_system.md")},
        {"role": "user", "content": user_message},
    ]


@observe(name="decide", capture_input=False)
def decide(record: TestCase, examples: list[TestCase]) -> BotDecision:
    """Ask the decider model for a structured decision about one record."""
    completion = get_openai_client().chat.completions.parse(
        model=DECIDER_MODEL,
        messages=build_messages(record, examples),
        response_format=BotDecision,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"Decider returned no parsed output for {record.task_id}: {message.refusal}")
    return message.parsed


@observe(name="run_bot", capture_input=False)
def run_bot(record: TestCase, index: ExampleIndex, top_k: int) -> BotRun:
    """Full bot pipeline for one record: retrieve similar examples, then decide.

    The record itself is always excluded from retrieval so a labelled record can never
    see its own answer.
    """
    start_time = time.perf_counter()
    retrieved = index.top_k(record, k=top_k, exclude_task_id=record.task_id)
    examples = [example for example, _similarity in retrieved]
    decision = decide(record, examples)
    latency_ms = (time.perf_counter() - start_time) * 1000

    return BotRun(
        decision=decision,
        retrieved_examples=[
            RetrievedExample(task_id=example.task_id, similarity=round(similarity, 4))
            for example, similarity in retrieved
        ],
        latency_ms=round(latency_ms, 1),
    )
