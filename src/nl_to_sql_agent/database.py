"""Read-only database access and schema description.

Two things live here:

* :class:`Database` -- a connection that is read-only *at the connection level*,
  not merely by convention. This is the control that actually makes a write
  impossible; the SQL guard layered on top of it is defence in depth.
* :func:`describe_schema` -- renders the allowed tables as compact DDL for the
  prompt. The model can only write correct SQL for a schema it has been shown,
  and showing it only the allowed tables keeps the prompt honest about what
  exists.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class DatabaseError(RuntimeError):
    """Raised when a query cannot be executed."""


@dataclass(frozen=True)
class QueryResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]


class Database:
    """A deliberately read-only SQLite connection.

    SQLite is opened with ``mode=ro`` via its URI syntax, so the engine itself
    rejects any write. That matters more than any application-level check: a
    bug in this program cannot turn into a modified database.

    In a server database the equivalent is a role granted ``SELECT`` and
    nothing else, on only the tables the agent is meant to see.
    """

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise DatabaseError(
                f"database not found: {self.path}. "
                "Run `python scripts/build_demo_db.py` first."
            )
        self._conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, timeout=timeout
        )
        # Stop a runaway query (a cartesian product, say) from hanging the
        # process: SQLite calls this handler periodically and a non-zero
        # return aborts the statement.
        self._budget = 0
        self._conn.set_progress_handler(self._interrupt_if_over_budget, 10_000)

    def _interrupt_if_over_budget(self) -> int:
        self._budget += 1
        return 1 if self._budget > 2_000 else 0

    def execute(self, sql: str, max_rows: int = 1000) -> QueryResult:
        """Run a read query and return at most ``max_rows`` rows."""
        self._budget = 0
        try:
            cur = self._conn.execute(sql)
        except sqlite3.OperationalError as exc:
            raise DatabaseError(str(exc)) from exc
        except sqlite3.DatabaseError as exc:
            raise DatabaseError(str(exc)) from exc

        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        cur.close()
        return QueryResult(columns, [tuple(r) for r in rows[:max_rows]], truncated)

    def table_names(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def columns_of(self, table: str) -> List[Tuple[str, str]]:
        # PRAGMA does not accept bound parameters, so the identifier is
        # validated against the real table list rather than interpolated blind.
        if table not in self.table_names():
            raise DatabaseError(f"no such table: {table}")
        rows = self._conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        return [(r[1], r[2]) for r in rows]

    def foreign_keys_of(self, table: str) -> List[Tuple[str, str, str]]:
        """Return (column, referenced_table, referenced_column) triples."""
        if table not in self.table_names():
            raise DatabaseError(f"no such table: {table}")
        rows = self._conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        return [(r[3], r[2], r[4]) for r in rows]

    def sample_values(self, table: str, column: str, limit: int = 5) -> List[Any]:
        """A few distinct values, to help the model match on real categories."""
        if table not in self.table_names():
            raise DatabaseError(f"no such table: {table}")
        valid = {c for c, _ in self.columns_of(table)}
        if column not in valid:
            raise DatabaseError(f"no such column: {table}.{column}")
        rows = self._conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL LIMIT {int(limit)}'
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def describe_schema(
    db: Database,
    tables: Iterable[str],
    *,
    sample_columns: Optional[Dict[str, Sequence[str]]] = None,
) -> str:
    """Render ``tables`` as compact DDL for the prompt.

    Only the named tables appear. A table the model is never shown is a table
    it will not reference -- which is a useful first line of defence, though
    emphatically not a sufficient one, since the model may still guess.
    """
    sample_columns = sample_columns or {}
    out: List[str] = []

    for table in tables:
        cols = db.columns_of(table)
        fks = {col: (ref_t, ref_c) for col, ref_t, ref_c in db.foreign_keys_of(table)}

        lines = [f"TABLE {table} ("]
        for i, (name, decl_type) in enumerate(cols):
            # The comma belongs to the column, before any trailing comment --
            # otherwise it lands inside the comment and reads as part of it.
            comma = "," if i < len(cols) - 1 else ""
            line = f"    {name} {decl_type or 'TEXT'}{comma}"
            if name in fks:
                ref_t, ref_c = fks[name]
                line += f"  -- references {ref_t}.{ref_c}"
            lines.append(line)
        lines.append(")")

        for column in sample_columns.get(table, ()):
            try:
                values = db.sample_values(table, column)
            except DatabaseError:
                continue
            if values:
                rendered = ", ".join(repr(v) for v in values)
                lines.append(f"-- {table}.{column} values include: {rendered}")

        out.append("\n".join(lines))

    return "\n\n".join(out)
