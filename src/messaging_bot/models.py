"""Pydantic contracts for the bot's input records, its output decision, and evaluation results.

Input models are permissive (unknown fields are kept) because hold-out records may
carry fields we have never seen. Output models are strict because they are sent to
OpenAI as the structured-output schema, which requires every field to be present
(optional values are expressed as nullable instead of defaulted).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Input: one test case from a JSONL file
# ---------------------------------------------------------------------------


class Consent(BaseModel):
    model_config = ConfigDict(extra="allow")

    email_opt_in: bool = False
    sms_opt_in: bool = False
    voice_opt_in: bool = False

    def allows_channel(self, channel: str) -> bool | None:
        """Return whether the person opted in to a channel, or None for unknown channels."""
        opt_in_flag = getattr(self, f"{channel.lower()}_opt_in", None)
        if opt_in_flag is None:
            return None
        return bool(opt_in_flag)


class ExpectedOutcome(BaseModel):
    """The labelled answer. Kept as loose dicts so new shapes in the hold-out set still load."""

    model_config = ConfigDict(extra="allow")

    next_message: dict[str, Any] | None = None
    next_action: dict[str, Any] | None = None


class TestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    persona: str | None = None
    lifecycle_stage: str | None = None
    consent: Consent | None = None
    channel_preferences: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    assertions: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    expected: ExpectedOutcome | None = None

    # Which file the record came from. Excluded from dumps so it never reaches a prompt.
    source_file: str = Field(default="", exclude=True)

    @property
    def constraints(self) -> dict[str, Any]:
        return self.assertions.get("constraints", {}) or {}


# ---------------------------------------------------------------------------
# Output: the bot's decision (OpenAI structured-output schema)
# ---------------------------------------------------------------------------


class CallToAction(BaseModel):
    type: str = Field(description="Machine-readable CTA type, e.g. schedule_tour.")
    options: list[str] | None = Field(description="Quick-reply options offered in the message, if any.")
    link: str | None = Field(description="URL the CTA points to, if any.")


class NextMessage(BaseModel):
    channel: str = Field(description="Delivery channel, e.g. sms, email or voice.")
    send_at: str = Field(description="ISO 8601 timestamp with the recipient's UTC offset.")
    subject: str | None = Field(description="Email subject line. Null for non-email channels.")
    body: str = Field(description="The full message text exactly as it would be sent.")
    cta: CallToAction | None = Field(description="Primary call to action, or null if none.")


class NextAction(BaseModel):
    type: str = Field(description="What the system should do next, e.g. start_cadence or follow_up_in_days.")
    name: str | None = Field(description="Name of the cadence or workflow, if the action has one.")
    value: int | None = Field(description="Numeric parameter for the action, e.g. days until follow-up.")


class BotDecision(BaseModel):
    reasoning: str = Field(
        description="Short explanation of why the bot chose to send or not, the channel, the timing and the content."
    )
    should_send: bool = Field(description="False when no message should be sent right now.")
    next_message: NextMessage | None = Field(description="The message to send. Null when should_send is false.")
    next_action: NextAction
    required_states_met: list[str] = Field(
        description="Which of the record's required_states this decision satisfies."
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Brief justification for the scores below.")
    body_semantic: float = Field(description="0-1: does the body convey the same meaning and intent as expected?")
    cta: float = Field(description="0-1: does the call to action match the expected type and options/link?")
    next_action: float = Field(description="0-1: does the next action match the expected follow-up plan?")
    personalization: float = Field(description="0-1: how well the message uses the recipient's own context.")
    overall: float = Field(description="0-1: overall semantic match with the expected outcome.")
    safety_violations: list[str] = Field(
        description="PII leaks, discriminatory language, missing required opt-out, or consent breaches. Empty if none."
    )


class RetrievedExample(BaseModel):
    task_id: str
    similarity: float


class BotRun(BaseModel):
    """Everything one bot invocation produced: the decision plus how it got there."""

    decision: BotDecision
    retrieved_examples: list[RetrievedExample]
    latency_ms: float


class CaseResult(BaseModel):
    task_id: str
    source_file: str
    retrieved_examples: list[RetrievedExample]
    decision: BotDecision
    checks: dict[str, bool | float | None]
    verdict: JudgeVerdict | None
    latency_ms: float
    trace_id: str | None = None


class EvalSummary(BaseModel):
    run_id: str
    scored_case_count: int
    failed_task_ids: dict[str, str]
    check_pass_rates: dict[str, float | None]
    mean_judge_scores: dict[str, float]
    total_safety_violations: int
    p95_latency_ms: float | None
    latency_threshold_ms: float | None
    results_path: str
