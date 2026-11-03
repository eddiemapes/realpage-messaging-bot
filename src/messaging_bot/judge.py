"""LLM-as-judge: grades how semantically close the bot's decision is to the expected outcome."""

import json

from langfuse import observe

from messaging_bot.agent import get_openai_client
from messaging_bot.config import JUDGE_MODEL, load_prompt
from messaging_bot.models import BotDecision, JudgeVerdict, TestCase
from messaging_bot.records import record_for_prompt


@observe(name="judge", capture_input=False)
def judge_decision(record: TestCase, decision: BotDecision) -> JudgeVerdict:
    """Score one decision against the record's expected outcome."""
    record_context = json.dumps(record_for_prompt(record), indent=2, ensure_ascii=False)
    expected_outcome = json.dumps(record.expected.model_dump(mode="json"), indent=2, ensure_ascii=False)
    bot_decision = json.dumps(decision.model_dump(mode="json"), indent=2, ensure_ascii=False)

    user_message = (
        f"## Input record\n```json\n{record_context}\n```\n\n"
        f"## Expected outcome\n```json\n{expected_outcome}\n```\n\n"
        f"## Bot decision\n```json\n{bot_decision}\n```"
    )

    completion = get_openai_client().chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": load_prompt("judge_system.md")},
            {"role": "user", "content": user_message},
        ],
        response_format=JudgeVerdict,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"Judge returned no parsed output for {record.task_id}: {message.refusal}")
    return message.parsed
