"""Command-line interface.

    nl-sql "which genre sells best?"      # live, needs ANTHROPIC_API_KEY
    nl-sql --offline --demo               # scripted answers, no key, no network
    nl-sql --offline --demo --show-sql    # with the pipeline trace
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from .agent import Agent, AgentAnswer
from .config import (
    ALLOWED_TABLES,
    DEFAULT_DB_PATH,
    MAX_LIMIT,
    OFF_LIMITS_TABLES,
    ROWS_TO_MODEL,
    SAMPLE_COLUMNS,
)
from .database import Database, DatabaseError
from .demo import DEMO_QUESTIONS, build_scripted_model
from .models import AnthropicModel, ModelError

DIM, BOLD, RED, GREEN, YELLOW, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[0m"
)


def _supports_colour(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


class Printer:
    def __init__(self, stream=None) -> None:
        # Keep `None` meaning "current sys.stdout" rather than binding the
        # stream now: binding at construction breaks redirection, and silently
        # writes to the pre-redirect stream.
        self._stream = stream

    @property
    def stream(self):
        return sys.stdout if self._stream is None else self._stream

    @property
    def colour(self) -> bool:
        return _supports_colour(self.stream)

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text

    def line(self, text: str = "") -> None:
        print(text, file=self.stream)

    def question(self, text: str) -> None:
        self.line()
        self.line(self._c(f"? {text}", BOLD))

    def trace(self, answer: AgentAnswer) -> None:
        for step in answer.steps:
            mark = "ok" if step.ok else "!!"
            colour = GREEN if step.ok else RED
            head = self._c(f"  [{mark}] {step.name:8}", colour)
            timing = self._c(f"{step.duration_ms:.0f}ms", DIM)
            lines = step.detail.splitlines() or [""]
            self.line(f"{head} {self._c(lines[0], DIM)}  {timing}")
            # Keep multi-line SQL aligned under the first line of the step.
            for extra in lines[1:]:
                self.line(f"{' ' * 15}{self._c(extra, DIM)}")

    def table(self, answer: AgentAnswer, limit: int = 10) -> None:
        result = answer.result
        if result is None or not result.rows:
            return
        widths = [len(c) for c in result.columns]
        shown = result.rows[:limit]
        for row in shown:
            for i, value in enumerate(row):
                widths[i] = max(widths[i], len(_fmt(value)))
        header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(result.columns))
        self.line("  " + self._c(header, BOLD))
        self.line("  " + self._c("-" * len(header), DIM))
        for row in shown:
            self.line("  " + "  ".join(
                _fmt(v).ljust(widths[i]) for i, v in enumerate(row)))
        if result.row_count > len(shown):
            self.line("  " + self._c(
                f"... {result.row_count - len(shown)} more row(s)", DIM))

    def answer(self, answer: AgentAnswer) -> None:
        if answer.refused:
            self.line(self._c(f"  {answer.refusal_reason}", YELLOW))
        elif answer.answer:
            self.line(f"  {answer.answer}")


def _fmt(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nl-sql",
        description="Ask a bookshop database questions in plain English.",
    )
    p.add_argument("question", nargs="*", help="the question to ask")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH),
                   help="path to the SQLite database")
    p.add_argument("--offline", action="store_true",
                   help="use scripted answers; no API key or network needed")
    p.add_argument("--demo", action="store_true",
                   help="run the built-in demo questions")
    p.add_argument("--show-sql", action="store_true",
                   help="print the pipeline trace, including generated SQL")
    p.add_argument("--schema", action="store_true",
                   help="print the schema the model is shown, then exit")
    p.add_argument("--max-rows", type=int, default=10,
                   help="rows to display (default: 10)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = Printer()

    try:
        db = Database(args.db)
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.schema:
        from .database import describe_schema
        out.line(describe_schema(db, ALLOWED_TABLES, sample_columns=SAMPLE_COLUMNS))
        out.line()
        out.line(f"Not readable by the agent: {', '.join(OFF_LIMITS_TABLES)}")
        return 0

    questions: List[str] = DEMO_QUESTIONS if args.demo else [" ".join(args.question)]
    if not questions or not questions[0]:
        build_parser().print_help()
        return 2

    try:
        model = build_scripted_model() if args.offline else AnthropicModel()
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agent = Agent(
        db, model,
        allowed_tables=ALLOWED_TABLES,
        max_limit=MAX_LIMIT,
        rows_to_model=ROWS_TO_MODEL,
        sample_columns=SAMPLE_COLUMNS,
    )

    refused = 0
    for question in questions:
        out.question(question)
        answer = agent.ask(question)
        if args.show_sql:
            out.trace(answer)
        if answer.ok:
            out.table(answer, limit=args.max_rows)
            out.line()
        out.answer(answer)
        if answer.refused:
            refused += 1

    out.line()
    db.close()
    # Refusals are a normal outcome, not a program failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
