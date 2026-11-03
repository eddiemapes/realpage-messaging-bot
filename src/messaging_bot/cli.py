"""Typer CLI for the messaging bot.

    uv run messaging-bot validate
    uv run messaging-bot similar <task_id>
    uv run messaging-bot decide <task_id>
    uv run messaging-bot eval [--only-file data/holdout.jsonl]
"""

import json
from pathlib import Path
from typing import Annotated

import typer
from langfuse import get_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from messaging_bot.agent import run_bot
from messaging_bot.config import DEFAULT_DATA_PATHS, DEFAULT_TOP_K
from messaging_bot.evaluate import run_leave_one_out
from messaging_bot.models import BotDecision, CaseResult, EvalSummary, TestCase
from messaging_bot.records import (
    RecordLoadError,
    find_record,
    load_records,
    load_records_from_file,
    retrieval_text,
    validate_records,
)
from messaging_bot.retrieval import ExampleIndex

app = typer.Typer(help="Context-aware messaging bot that learns from example records.", no_args_is_help=True)
console = Console()

DataOption = Annotated[
    list[Path] | None,
    typer.Option("--data", "-d", help="Data file(s) to pool. Defaults to data/sample.jsonl + data/holdout.jsonl."),
]
TopKOption = Annotated[int, typer.Option("--top-k", "-k", help="How many similar examples to retrieve.")]


def load_or_exit(data_paths: list[Path] | None) -> list[TestCase]:
    try:
        return load_records(data_paths or DEFAULT_DATA_PATHS)
    except RecordLoadError as error:
        console.print(f"[bold red]Could not load records:[/] {error}")
        raise typer.Exit(code=1)


def format_check(value: bool | float | None) -> str:
    if value is None:
        return "[dim]-[/]"
    if isinstance(value, bool):
        return "[green]pass[/]" if value else "[red]FAIL[/]"
    return f"{value:g}"


def to_json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False) if value else ""


def find_record_or_exit(records: list[TestCase], task_id: str) -> TestCase:
    try:
        return find_record(records, task_id)
    except KeyError as error:
        console.print(f"[bold red]{error.args[0]}[/]")
        raise typer.Exit(code=1)


def format_score(value: float) -> str:
    colour = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
    return f"[{colour}]{value:.2f}[/]"


def print_decision_vs_expected(record: TestCase, decision: BotDecision) -> None:
    expected_message = (record.expected.next_message if record.expected else None) or {}
    expected_action = (record.expected.next_action if record.expected else None) or {}
    bot_message = decision.next_message

    table = Table(title=f"Decision for {record.task_id}", show_lines=True, expand=True)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Bot")
    table.add_column("Expected" if record.expected else "Expected (none)")

    table.add_row("should_send", str(decision.should_send), str(bool(expected_message)) if record.expected else "")
    table.add_row("channel", bot_message.channel if bot_message else "", str(expected_message.get("channel", "")))
    table.add_row("send_at", bot_message.send_at if bot_message else "", str(expected_message.get("send_at", "")))
    table.add_row("subject", (bot_message.subject or "") if bot_message else "", str(expected_message.get("subject") or ""))
    table.add_row("body", bot_message.body if bot_message else "", str(expected_message.get("body", "")))
    table.add_row(
        "cta",
        to_json_text(bot_message.cta.model_dump(exclude_none=True)) if bot_message and bot_message.cta else "",
        to_json_text(expected_message.get("cta")),
    )
    table.add_row(
        "next_action",
        to_json_text(decision.next_action.model_dump(exclude_none=True)),
        to_json_text(expected_action),
    )
    console.print(table)
    console.print(Panel(decision.reasoning, title="Bot reasoning", border_style="cyan"))


def print_case_line(result: CaseResult) -> None:
    overall = format_score(result.verdict.overall) if result.verdict else "-"
    send_ok = format_check(result.checks.get("send_decision_match"))
    console.print(f"  [bold]{result.task_id}[/]  send:{send_ok}  judge:{overall}  {result.latency_ms:.0f} ms")


def print_case_failed(task_id: str, error: Exception) -> None:
    console.print(f"  [bold]{task_id}[/]  [red]ERROR[/] {error}")


def print_results_table(results: list[CaseResult]) -> None:
    table = Table(title="Leave-one-out results", show_lines=False)
    table.add_column("task_id", style="bold")
    table.add_column("file", style="dim")
    for column_name in ["send", "channel", "consent", "opt-out", "date", "hrs off", "cta", "action"]:
        table.add_column(column_name, justify="center")
    table.add_column("judge", justify="right")
    table.add_column("ms", justify="right")

    for result in results:
        checks = result.checks
        table.add_row(
            result.task_id,
            result.source_file,
            format_check(checks.get("send_decision_match")),
            format_check(checks.get("channel_match")),
            format_check(checks.get("consent_respected")),
            format_check(checks.get("opt_out_present")),
            format_check(checks.get("send_at_date_match")),
            format_check(checks.get("send_at_hours_off")),
            format_check(checks.get("cta_type_match")),
            format_check(checks.get("next_action_type_match")),
            format_score(result.verdict.overall) if result.verdict else "-",
            f"{result.latency_ms:.0f}",
        )
    console.print(table)


def print_summary(summary: EvalSummary) -> None:
    table = Table(title=f"Summary ({summary.run_id})", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Scored cases", str(summary.scored_case_count))
    if summary.failed_task_ids:
        table.add_row("Failed cases", "[red]" + ", ".join(summary.failed_task_ids) + "[/]")
    for check_name, pass_rate in summary.check_pass_rates.items():
        table.add_row(f"check: {check_name}", "-" if pass_rate is None else f"{pass_rate:.0%}")
    for score_name, mean_score in summary.mean_judge_scores.items():
        table.add_row(f"judge: {score_name}", format_score(mean_score))
    table.add_row("Safety violations", str(summary.total_safety_violations))
    if summary.p95_latency_ms is not None:
        latency_text = f"{summary.p95_latency_ms:.0f} ms"
        if summary.latency_threshold_ms is not None:
            within_budget = summary.p95_latency_ms <= summary.latency_threshold_ms
            latency_text += f" (target {summary.latency_threshold_ms:.0f} ms: {'[green]ok[/]' if within_budget else '[red]over[/]'})"
        table.add_row("p95 latency", latency_text)
    table.add_row("Results file", summary.results_path)
    console.print(table)


@app.command()
def validate(data: DataOption = None) -> None:
    """Check data files parse cleanly and are usable for retrieval and evaluation."""
    data_paths = data or DEFAULT_DATA_PATHS

    file_table = Table(title="Data files")
    file_table.add_column("File")
    file_table.add_column("Records", justify="right")
    file_table.add_column("Labelled", justify="right")
    all_records: list[TestCase] = []
    for path in data_paths:
        try:
            file_records = load_records_from_file(path)
        except RecordLoadError as error:
            console.print(f"[bold red]{error}[/]")
            raise typer.Exit(code=1)
        status = str(len(file_records)) if path.exists() else "[yellow]missing[/]"
        labelled_count = sum(1 for record in file_records if record.expected is not None)
        file_table.add_row(str(path.name), status, str(labelled_count))
        all_records.extend(file_records)
    console.print(file_table)

    problems = validate_records(all_records)
    if problems:
        for problem in problems:
            console.print(f"[yellow]![/] {problem}")
    else:
        console.print(f"[green]All {len(all_records)} records look good.[/]")


@app.command()
def similar(
    task_id: Annotated[str, typer.Argument(help="Record to find neighbours for.")],
    data: DataOption = None,
    top_k: TopKOption = DEFAULT_TOP_K,
    show_text: Annotated[bool, typer.Option(help="Print the text that gets embedded.")] = False,
) -> None:
    """Show which examples retrieval would hand the bot for a record (itself excluded)."""
    records = load_or_exit(data)
    record = find_record_or_exit(records, task_id)
    index = ExampleIndex(records)
    try:
        neighbours = index.top_k(record, k=top_k, exclude_task_id=record.task_id)
    finally:
        get_client().flush()

    if show_text:
        console.print(Panel(retrieval_text(record), title=f"Embedded text: {task_id}"))

    table = Table(title=f"Top {top_k} examples for {task_id}")
    table.add_column("Rank", justify="right")
    table.add_column("task_id", style="bold")
    table.add_column("file", style="dim")
    table.add_column("Cosine", justify="right")
    for rank, (example, similarity) in enumerate(neighbours, start=1):
        table.add_row(str(rank), example.task_id, example.source_file, f"{similarity:.3f}")
    console.print(table)
    if not neighbours:
        console.print("[yellow]No other labelled records to retrieve from.[/]")


@app.command()
def decide(
    task_id: Annotated[str, typer.Argument(help="Record to run the bot on.")],
    data: DataOption = None,
    top_k: TopKOption = DEFAULT_TOP_K,
) -> None:
    """Run the bot on one record and compare its decision with the expected outcome."""
    records = load_or_exit(data)
    record = find_record_or_exit(records, task_id)
    index = ExampleIndex(records)
    try:
        bot_run = run_bot(record, index, top_k)
    finally:
        get_client().flush()

    retrieved_ids = ", ".join(f"{example.task_id} ({example.similarity:.2f})" for example in bot_run.retrieved_examples)
    console.print(f"[dim]Examples used: {retrieved_ids or 'none'} | {bot_run.latency_ms:.0f} ms[/]")
    print_decision_vs_expected(record, bot_run.decision)


@app.command(name="eval")
def evaluate(
    data: DataOption = None,
    top_k: TopKOption = DEFAULT_TOP_K,
    only_file: Annotated[
        Path | None,
        typer.Option(help="Only score records from this file (retrieval still uses the whole pool)."),
    ] = None,
    task_id: Annotated[list[str] | None, typer.Option(help="Only score these task ids. Repeatable.")] = None,
) -> None:
    """Leave-one-out evaluation over the pooled records, scored by rule checks + LLM judge."""
    records = load_or_exit(data)

    target_task_ids: set[str] | None = None
    if only_file is not None:
        target_task_ids = {record.task_id for record in records if record.source_file == only_file.name}
    if task_id:
        requested_ids = set(task_id)
        target_task_ids = requested_ids if target_task_ids is None else target_task_ids & requested_ids
    if target_task_ids is not None and not target_task_ids:
        console.print("[yellow]No records match the requested filters.[/]")
        raise typer.Exit(code=1)

    labelled_count = sum(1 for record in records if record.expected is not None)
    console.print(f"Pool: {labelled_count} labelled records | top_k={top_k}")
    try:
        results, summary = run_leave_one_out(
            records,
            top_k=top_k,
            target_task_ids=target_task_ids,
            on_case_complete=print_case_line,
            on_case_failed=print_case_failed,
        )
    finally:
        get_client().flush()

    print_results_table(results)
    print_summary(summary)


if __name__ == "__main__":
    app()
