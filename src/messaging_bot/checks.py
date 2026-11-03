"""Deterministic rule checks on a decision.

Each check returns True (pass), False (fail) or None (not applicable to this record).
Message-level comparisons only run when both the expected outcome and the bot send a
message; a send/no-send mismatch is captured once by `send_decision_match`.
"""

import re
from datetime import datetime

from messaging_bot.models import BotDecision, TestCase

OPT_OUT_PATTERN = re.compile(r"\b(stop|opt[\s-]?out|unsubscribe)\b", re.IGNORECASE)


def _parse_timestamp(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return None


def check_send_decision_match(record: TestCase, decision: BotDecision) -> bool | None:
    if record.expected is None:
        return None
    expected_sends = record.expected.next_message is not None
    return expected_sends == decision.should_send


def check_consent_respected(record: TestCase, decision: BotDecision) -> bool | None:
    if not decision.should_send or decision.next_message is None:
        return True
    if record.consent is None:
        return None
    return record.consent.allows_channel(decision.next_message.channel)


def check_opt_out_present(record: TestCase, decision: BotDecision) -> bool | None:
    if not record.constraints.get("include_opt_out_instructions"):
        return None
    if decision.next_message is None:
        return None
    return bool(OPT_OUT_PATTERN.search(decision.next_message.body))


def check_subject_rule(decision: BotDecision) -> bool | None:
    """Emails need a subject line; other channels should not have one."""
    if decision.next_message is None:
        return None
    has_subject = bool(decision.next_message.subject and decision.next_message.subject.strip())
    if decision.next_message.channel.lower() == "email":
        return has_subject
    return not has_subject


def check_channel_match(expected_message: dict, decision: BotDecision) -> bool:
    expected_channel = str(expected_message.get("channel", "")).lower()
    return decision.next_message.channel.lower() == expected_channel


def check_send_at_date_match(expected_message: dict, decision: BotDecision) -> bool | None:
    """Same calendar day in the expected timestamp's timezone."""
    expected_time = _parse_timestamp(expected_message.get("send_at"))
    actual_time = _parse_timestamp(decision.next_message.send_at)
    if expected_time is None or actual_time is None:
        return None
    if expected_time.tzinfo is None or actual_time.tzinfo is None:
        return expected_time.date() == actual_time.date()
    return actual_time.astimezone(expected_time.tzinfo).date() == expected_time.date()


def send_at_hours_off(expected_message: dict, decision: BotDecision) -> float | None:
    expected_time = _parse_timestamp(expected_message.get("send_at"))
    actual_time = _parse_timestamp(decision.next_message.send_at)
    if expected_time is None or actual_time is None:
        return None
    if (expected_time.tzinfo is None) != (actual_time.tzinfo is None):
        return None
    return round(abs((actual_time - expected_time).total_seconds()) / 3600, 2)


def check_cta_type_match(expected_message: dict, decision: BotDecision) -> bool | None:
    expected_cta = expected_message.get("cta")
    actual_cta = decision.next_message.cta
    if not expected_cta and actual_cta is None:
        return True
    if not expected_cta or actual_cta is None:
        return False
    return str(expected_cta.get("type", "")).lower() == actual_cta.type.lower()


def check_next_action_type_match(record: TestCase, decision: BotDecision) -> bool | None:
    if record.expected is None or not record.expected.next_action:
        return None
    expected_type = str(record.expected.next_action.get("type", "")).lower()
    return decision.next_action.type.lower() == expected_type


def run_checks(record: TestCase, decision: BotDecision) -> dict[str, bool | float | None]:
    """Run every rule check and return them keyed by name."""
    checks: dict[str, bool | float | None] = {
        "send_decision_match": check_send_decision_match(record, decision),
        "consent_respected": check_consent_respected(record, decision),
        "opt_out_present": check_opt_out_present(record, decision),
        "subject_rule": check_subject_rule(decision),
        "channel_match": None,
        "send_at_date_match": None,
        "send_at_hours_off": None,
        "cta_type_match": None,
        "next_action_type_match": check_next_action_type_match(record, decision),
    }

    expected_message = record.expected.next_message if record.expected else None
    if expected_message and decision.next_message is not None:
        checks["channel_match"] = check_channel_match(expected_message, decision)
        checks["send_at_date_match"] = check_send_at_date_match(expected_message, decision)
        checks["send_at_hours_off"] = send_at_hours_off(expected_message, decision)
        checks["cta_type_match"] = check_cta_type_match(expected_message, decision)

    return checks
