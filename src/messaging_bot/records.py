"""Loading, validating and rendering test-case records.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from messaging_bot.models import TestCase

# Identity fields carry no signal about *which situation* a record describes,
# so they are left out of the text used for similarity search.
KEYS_IGNORED_FOR_RETRIEVAL = {"first_name", "last_name", "full_name", "name", "email", "phone"}


class RecordLoadError(Exception):
    """Raised when a data file cannot be parsed into test cases."""


def _line_number_at(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def parse_json_objects(text: str, file_label: str) -> list[dict[str, Any]]:
    """Read every top-level JSON value in `text`, regardless of line layout."""
    decoder = json.JSONDecoder()
    parsed_objects: list[dict[str, Any]] = []
    position = 0

    while position < len(text):
        # Skip whitespace and stray commas between pasted objects.
        if text[position].isspace() or text[position] == ",":
            position += 1
            continue

        try:
            value, end_position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            line_number = _line_number_at(text, error.pos)
            raise RecordLoadError(f"{file_label}, line {line_number}: invalid JSON ({error.msg})") from error

        line_number = _line_number_at(text, position)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict):
                raise RecordLoadError(f"{file_label}, line {line_number}: expected a JSON object, got {type(item).__name__}")
            parsed_objects.append(item)

        position = end_position

    return parsed_objects


def load_records_from_file(path: Path) -> list[TestCase]:
    """Load one data file. A missing or empty file yields no records."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8-sig")
    records: list[TestCase] = []
    for index, raw_record in enumerate(parse_json_objects(text, path.name), start=1):
        try:
            record = TestCase.model_validate(raw_record)
        except ValidationError as error:
            task_id = raw_record.get("task_id", f"record #{index}")
            raise RecordLoadError(f"{path.name}, {task_id}: {error}") from error
        record.source_file = path.name
        records.append(record)
    return records


def load_records(paths: list[Path]) -> list[TestCase]:
    """Load and concatenate records from several files, in order."""
    records: list[TestCase] = []
    for path in paths:
        records.extend(load_records_from_file(path))
    return records


def find_record(records: list[TestCase], task_id: str) -> TestCase:
    for record in records:
        if record.task_id == task_id:
            return record
    known_ids = ", ".join(record.task_id for record in records)
    raise KeyError(f"No record with task_id '{task_id}'. Known ids: {known_ids}")


def validate_records(records: list[TestCase]) -> list[str]:
    """Return human-readable problems that would hurt retrieval or evaluation."""
    problems: list[str] = []

    task_id_counts = Counter(record.task_id for record in records)
    for task_id, count in task_id_counts.items():
        if count > 1:
            problems.append(f"Duplicate task_id '{task_id}' appears {count} times (leave-one-out would leak the answer).")

    for record in records:
        label = f"{record.source_file}: {record.task_id}"
        if record.expected is None:
            problems.append(f"{label} has no 'expected' block: it can be run, but not scored or used as an example.")
        if record.consent is None:
            problems.append(f"{label} has no 'consent' block: consent checks will be skipped.")
        if not record.input:
            problems.append(f"{label} has an empty 'input' block.")

    return problems


def record_for_prompt(record: TestCase) -> dict[str, Any]:
    """The part of a record the bot is allowed to see.

    Drops `expected` (the answer), `task_id` (its name describes the scenario and would
    leak the label) and `thresholds` (evaluation targets, not decision context).
    """
    return record.model_dump(mode="json", exclude={"task_id", "expected", "thresholds"})


def _flatten_to_lines(value: Any, key_prefix: str, lines: list[str]) -> None:
    if isinstance(value, dict):
        for key, child_value in value.items():
            if key in KEYS_IGNORED_FOR_RETRIEVAL:
                continue
            child_prefix = f"{key_prefix}.{key}" if key_prefix else key
            _flatten_to_lines(child_value, child_prefix, lines)
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            lines.append(f"{key_prefix}: {', '.join(str(item) for item in value)}")
        else:
            for item in value:
                _flatten_to_lines(item, key_prefix, lines)
    elif value is not None:
        lines.append(f"{key_prefix}: {value}")


def retrieval_text(record: TestCase) -> str:
    """Compact `key: value` lines describing the record's situation, used for embeddings.

    Raw JSON would blow past MiniLM's 256-token window, so this keeps only the
    situational fields in a short, readable form.
    """
    situation = {
        "persona": record.persona,
        "lifecycle_stage": record.lifecycle_stage,
        "consent": record.consent.model_dump() if record.consent else None,
        "channel_preferences": record.channel_preferences,
        "input": record.input,
        "constraints": record.constraints,
    }
    extra_fields = record.model_extra or {}
    situation.update(extra_fields)

    lines: list[str] = []
    _flatten_to_lines(situation, "", lines)
    return "\n".join(lines)
