"""Leave-one-out evaluation: every labelled record is decided using all *other* records
as retrievable examples, then scored by rule checks and the LLM judge.

Each case is one Langfuse trace (retrieve -> decide -> judge) with all scores attached,
and every trace in a run shares one session id so the run can be filtered as a group.
"""

import json
from collections.abc import Callable
from datetime import datetime

import numpy as np
from langfuse import get_client, observe, propagate_attributes

from messaging_bot.agent import run_bot
from messaging_bot.checks import run_checks
from messaging_bot.config import OUTPUTS_DIR
from messaging_bot.judge import judge_decision
from messaging_bot.models import CaseResult, EvalSummary, TestCase
from messaging_bot.retrieval import ExampleIndex

JUDGE_SCORE_NAMES = ["body_semantic", "cta", "next_action", "personalization", "overall"]


def attach_scores_to_trace(checks: dict[str, bool | float | None], judge_scores: dict[str, float]) -> None:
    """Record every check and judge score on the current Langfuse trace."""
    langfuse = get_client()
    for check_name, check_value in checks.items():
        if check_value is None:
            continue
        if isinstance(check_value, bool):
            langfuse.score_current_trace(name=check_name, value=float(check_value), data_type="BOOLEAN")
        else:
            langfuse.score_current_trace(name=check_name, value=float(check_value), data_type="NUMERIC")
    for score_name, score_value in judge_scores.items():
        langfuse.score_current_trace(name=f"judge_{score_name}", value=score_value, data_type="NUMERIC")


@observe(name="eval_case", capture_input=False, capture_output=False)
def evaluate_case(record: TestCase, index: ExampleIndex, top_k: int) -> CaseResult:
    """Run the bot on one labelled record and score the result."""
    bot_run = run_bot(record, index, top_k)
    checks = run_checks(record, bot_run.decision)
    verdict = judge_decision(record, bot_run.decision)

    personalization_minimum = record.thresholds.get("personalization_score_min")
    if personalization_minimum is not None:
        checks["personalization_meets_threshold"] = verdict.personalization >= float(personalization_minimum)
    checks["no_safety_violations"] = len(verdict.safety_violations) == 0

    judge_scores = {score_name: getattr(verdict, score_name) for score_name in JUDGE_SCORE_NAMES}
    attach_scores_to_trace(checks, judge_scores)

    langfuse = get_client()
    langfuse.update_current_span(
        input={"task_id": record.task_id, "source_file": record.source_file},
        output={"decision": bot_run.decision.model_dump(mode="json"), "verdict": verdict.model_dump(mode="json")},
    )

    return CaseResult(
        task_id=record.task_id,
        source_file=record.source_file,
        retrieved_examples=bot_run.retrieved_examples,
        decision=bot_run.decision,
        checks=checks,
        verdict=verdict,
        latency_ms=bot_run.latency_ms,
        trace_id=langfuse.get_current_trace_id(),
    )


def summarize_results(
    run_id: str,
    results: list[CaseResult],
    failed_task_ids: dict[str, str],
    targets: list[TestCase],
    results_path: str,
) -> EvalSummary:
    """Aggregate per-case results into pass rates, mean judge scores and latency stats."""
    check_names = sorted({check_name for result in results for check_name in result.checks})
    check_pass_rates: dict[str, float | None] = {}
    for check_name in check_names:
        if check_name == "send_at_hours_off":
            continue
        applicable_values = [
            result.checks[check_name] for result in results if isinstance(result.checks.get(check_name), bool)
        ]
        check_pass_rates[check_name] = (
            round(sum(applicable_values) / len(applicable_values), 3) if applicable_values else None
        )

    mean_judge_scores: dict[str, float] = {}
    verdicts = [result.verdict for result in results if result.verdict is not None]
    for score_name in JUDGE_SCORE_NAMES:
        if verdicts:
            mean_judge_scores[score_name] = round(float(np.mean([getattr(verdict, score_name) for verdict in verdicts])), 3)

    latencies = [result.latency_ms for result in results]
    latency_thresholds = [
        float(record.thresholds["p95_latency_ms"]) for record in targets if "p95_latency_ms" in record.thresholds
    ]

    return EvalSummary(
        run_id=run_id,
        scored_case_count=len(results),
        failed_task_ids=failed_task_ids,
        check_pass_rates=check_pass_rates,
        mean_judge_scores=mean_judge_scores,
        total_safety_violations=sum(len(verdict.safety_violations) for verdict in verdicts),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 1) if latencies else None,
        latency_threshold_ms=min(latency_thresholds) if latency_thresholds else None,
        results_path=results_path,
    )


def run_leave_one_out(
    records: list[TestCase],
    top_k: int,
    target_task_ids: set[str] | None = None,
    on_case_complete: Callable[[CaseResult], None] | None = None,
    on_case_failed: Callable[[str, Exception], None] | None = None,
) -> tuple[list[CaseResult], EvalSummary]:
    """Evaluate labelled records against a retrieval index built from the whole pool.

    `target_task_ids` limits which records are scored (e.g. only the hold-out file);
    retrieval still draws from every labelled record except the one being scored.
    """
    run_id = datetime.now().strftime("eval-%Y%m%d-%H%M%S")
    index = ExampleIndex(records)

    targets = [record for record in records if record.expected is not None]
    if target_task_ids is not None:
        targets = [record for record in targets if record.task_id in target_task_ids]

    results: list[CaseResult] = []
    failed_task_ids: dict[str, str] = {}
    for record in targets:
        try:
            with propagate_attributes(
                session_id=run_id,
                trace_name=f"eval:{record.task_id}",
                tags=["leave-one-out", record.source_file],
                metadata={"task_id": record.task_id},
            ):
                result = evaluate_case(record, index, top_k)
        except Exception as error:  # keep the run going so one bad record doesn't sink a demo
            failed_task_ids[record.task_id] = str(error)
            if on_case_failed:
                on_case_failed(record.task_id, error)
            continue
        results.append(result)
        if on_case_complete:
            on_case_complete(result)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUTS_DIR / f"{run_id}.jsonl"
    with results_path.open("w", encoding="utf-8") as results_file:
        for result in results:
            results_file.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")

    summary = summarize_results(run_id, results, failed_task_ids, targets, str(results_path))
    return results, summary
