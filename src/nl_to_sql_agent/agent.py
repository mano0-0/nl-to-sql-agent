"""The question-to-answer pipeline.

    question -> generate SQL -> GUARD -> execute (read-only) -> explain

The guard step is not advice the model may decline to take. The generated SQL
is validated by `llm_sql_guard` before the database is contacted at all, and a
rejected query is reported to the user instead of being executed.

Why both a guard and a read-only connection: the connection is what makes a
write *impossible*; the guard is what stops a legal read from touching tables
outside the allowlist, returning unbounded rows, or calling functions like
`pg_read_file`. Each catches things the other cannot. See README.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_sql_guard import GuardResult, Policy, check

from .database import Database, DatabaseError, QueryResult, describe_schema
from .models import LanguageModel, ModelError, extract_sql

SQL_SYSTEM_PROMPT = """\
You translate questions about a bookshop database into a single SQL query.

Rules:
- Return ONE SQLite SELECT statement. No prose, no explanation, no semicolon.
- Use only the tables and columns in the schema below. Never invent a name.
- Always include a LIMIT of at most {max_limit}.
- Prefer explicit JOINs over subqueries where both work.
- When the question implies money, compute it as quantity * unit_price.
- If the question cannot be answered from this schema, return exactly:
  CANNOT_ANSWER

Schema:
{schema}
"""

ANSWER_SYSTEM_PROMPT = """\
You explain query results to someone who did not write the query and does not
read SQL.

- Answer the question directly in one or two sentences. Lead with the figure.
- Use the numbers given. Never estimate, extrapolate, or invent a value.
- If the result set is empty, say plainly that nothing matched.
- You are told the true row count; if fewer rows are shown than that count,
  say the answer is based on the first N.
- Do not describe the SQL or mention tables unless asked.
"""

CANNOT_ANSWER = "CANNOT_ANSWER"


@dataclass
class Step:
    """One stage of the pipeline, for display and for tests."""

    name: str
    detail: str
    ok: bool = True
    duration_ms: float = 0.0


@dataclass
class AgentAnswer:
    """Everything the pipeline produced, successful or not."""

    question: str
    answer: Optional[str] = None
    sql: Optional[str] = None
    result: Optional[QueryResult] = None
    guard: Optional[GuardResult] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    steps: List[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.answer is not None and not self.refused


class Agent:
    """Answers questions about one database, within one policy."""

    def __init__(
        self,
        db: Database,
        model: LanguageModel,
        *,
        allowed_tables: Sequence[str],
        max_limit: int = 200,
        rows_to_model: int = 40,
        sample_columns: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self.db = db
        self.model = model
        self.allowed_tables = list(allowed_tables)
        self.rows_to_model = rows_to_model
        self.policy = Policy(
            allowed_tables=set(allowed_tables),
            max_limit=max_limit,
            require_limit=True,
            allow_set_operations=False,
            dialect="sqlite",
        )
        self._schema = describe_schema(
            db, self.allowed_tables, sample_columns=sample_columns
        )

    # -- the pipeline -----------------------------------------------------

    def ask(self, question: str) -> AgentAnswer:
        answer = AgentAnswer(question=question)

        sql = self._generate_sql(question, answer)
        if sql is None:
            return answer

        if not self._guard(sql, answer):
            return answer

        result = self._execute(sql, answer)
        if result is None:
            return answer

        self._explain(question, result, answer)
        return answer

    # -- stages -----------------------------------------------------------

    def _generate_sql(self, question: str, answer: AgentAnswer) -> Optional[str]:
        started = time.perf_counter()
        system = SQL_SYSTEM_PROMPT.format(
            schema=self._schema, max_limit=self.policy.max_limit
        )
        try:
            raw = self.model.complete(system, question)
        except ModelError as exc:
            # Surface model failures honestly. Reporting an outage as "I can't
            # answer that" trains users to distrust correct refusals too.
            answer.refused = True
            answer.refusal_reason = str(exc)
            answer.steps.append(Step("generate", str(exc), ok=False,
                                     duration_ms=self._ms(started)))
            return None

        sql = extract_sql(raw)
        if not sql or sql.upper().startswith(CANNOT_ANSWER):
            answer.refused = True
            answer.refusal_reason = (
                "That question cannot be answered from the data available."
            )
            answer.steps.append(Step("generate", "model declined", ok=False,
                                     duration_ms=self._ms(started)))
            return None

        answer.sql = sql
        answer.steps.append(Step("generate", sql, duration_ms=self._ms(started)))
        return sql

    def _guard(self, sql: str, answer: AgentAnswer) -> bool:
        started = time.perf_counter()
        result = check(sql, self.policy)
        answer.guard = result

        if not result.ok:
            reasons = "; ".join(result.reasons)
            answer.refused = True
            answer.refusal_reason = f"The generated query was rejected: {reasons}"
            answer.steps.append(Step("guard", reasons, ok=False,
                                     duration_ms=self._ms(started)))
            return False

        tables = ", ".join(sorted(result.tables)) or "none"
        answer.steps.append(Step("guard", f"passed; reads {tables}",
                                 duration_ms=self._ms(started)))
        return True

    def _execute(self, sql: str, answer: AgentAnswer) -> Optional[QueryResult]:
        started = time.perf_counter()
        try:
            result = self.db.execute(sql, max_rows=self.policy.max_limit)
        except DatabaseError as exc:
            answer.refused = True
            answer.refusal_reason = f"The query could not be run: {exc}"
            answer.steps.append(Step("execute", str(exc), ok=False,
                                     duration_ms=self._ms(started)))
            return None

        answer.result = result
        answer.steps.append(
            Step("execute", f"{result.row_count} row(s)",
                 duration_ms=self._ms(started))
        )
        return result

    def _explain(
        self, question: str, result: QueryResult, answer: AgentAnswer
    ) -> None:
        started = time.perf_counter()
        prompt = self._render_for_model(question, result)
        try:
            text = self.model.complete(ANSWER_SYSTEM_PROMPT, prompt)
        except ModelError as exc:
            answer.refused = True
            answer.refusal_reason = str(exc)
            answer.steps.append(Step("explain", str(exc), ok=False,
                                     duration_ms=self._ms(started)))
            return

        answer.answer = text.strip()
        answer.steps.append(Step("explain", "answered",
                                 duration_ms=self._ms(started)))

    # -- helpers ----------------------------------------------------------

    def _render_for_model(self, question: str, result: QueryResult) -> str:
        """Render results compactly, sending a sample but the true row count.

        Sending every row of a large result wastes tokens and can push the
        question out of the window. Sending a sample *without* the real count
        is worse: the model then describes 40 rows as if they were all of them.
        """
        shown = result.rows[: self.rows_to_model]
        header = " | ".join(result.columns)
        body = "\n".join(
            " | ".join("NULL" if v is None else str(v) for v in row)
            for row in shown
        )
        note = ""
        if len(shown) < result.row_count:
            note = (
                f"\n(showing the first {len(shown)} of {result.row_count} rows)"
            )
        elif result.truncated:
            note = (
                f"\n(the query matched more than {result.row_count} rows; "
                "these are the first of them)"
            )

        return (
            f"Question: {question}\n\n"
            f"Rows returned: {result.row_count}\n\n"
            f"{header}\n{body}{note}"
        )

    @staticmethod
    def _ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000.0
