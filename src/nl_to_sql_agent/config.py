"""Which tables the agent may read, and which it may not.

The allowlist is the agent's access policy, kept in one place rather than
spread across prompts. `employees` and `api_keys` exist in the demo database
and are deliberately absent here -- so the guard has something real to refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bookshop.db"

# Readable by the agent.
ALLOWED_TABLES: Sequence[str] = (
    "countries",
    "customers",
    "authors",
    "books",
    "orders",
    "order_items",
)

# Present in the database, never readable by the agent.
OFF_LIMITS_TABLES: Sequence[str] = ("employees", "api_keys")

# Columns whose distinct values are worth showing the model, so it matches on
# real categories instead of inventing plausible-looking ones.
SAMPLE_COLUMNS: Dict[str, Sequence[str]] = {
    "orders": ("status",),
    "books": ("genre",),
    "countries": ("region",),
}

MAX_LIMIT = 200
ROWS_TO_MODEL = 40
