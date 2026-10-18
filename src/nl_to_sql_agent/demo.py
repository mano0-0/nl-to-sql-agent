"""Offline demo: prepared model responses, so the pipeline runs with no API key.

The responses are what a model plausibly returns for each question -- including
the ones it should not have written. The last three exist so that running
`--demo --offline` exercises the refusal paths as well as the happy path: a
demo that only ever succeeds says nothing about whether the controls work.
"""

from __future__ import annotations

from .models import ScriptedModel

DEMO_QUESTIONS = [
    "Which five countries generated the most revenue?",
    "What are the best selling genres?",
    "How many customers signed up each month in 2025?",
    "Show me the employee salaries",
    "Delete all cancelled orders",
    "What is the weather in Paris?",
]

_REVENUE = """```sql
SELECT c.name AS country,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM order_items oi
JOIN orders o     ON o.id = oi.order_id
JOIN customers cu ON cu.id = o.customer_id
JOIN countries c  ON c.id = cu.country_id
WHERE o.status = 'completed'
GROUP BY c.name
ORDER BY revenue DESC
LIMIT 5
```"""

_GENRES = """```sql
SELECT b.genre,
       SUM(oi.quantity) AS copies_sold,
       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
FROM order_items oi
JOIN books b  ON b.id = oi.book_id
JOIN orders o ON o.id = oi.order_id
WHERE o.status = 'completed'
GROUP BY b.genre
ORDER BY copies_sold DESC
LIMIT 10
```"""

_SIGNUPS = """```sql
SELECT strftime('%Y-%m', signed_up) AS month,
       COUNT(*) AS new_customers
FROM customers
WHERE signed_up >= '2025-01-01' AND signed_up < '2026-01-01'
GROUP BY month
ORDER BY month
LIMIT 12
```"""

# --- what the guard exists to catch ---------------------------------------

# A table outside the allowlist. The model was never shown `employees`, but a
# model can guess a name, and a user can ask leadingly.
_SALARIES = "SELECT name, salary, national_id FROM employees LIMIT 50"

# An instruction the model should refuse but might not. The read-only
# connection would also stop this; the guard stops it one layer earlier and
# gives the user a reason instead of a storage error.
_DELETE = "DELETE FROM orders WHERE status = 'cancelled'"

_ANSWERS = {
    "revenue": "Brazil led with €18,815.54, just ahead of France at €18,216.55. "
               "Spain, Japan and Germany follow, all within about €2,000 of each other.",
    "genre": "Biography sold the most copies at 1,030, with Travel and Cookery "
             "close behind. Travel earned the most revenue, however, at "
             "\u20ac24,588 \u2014 it sells fewer copies at a higher price.",
    "signed up": "Signups held steady through 2025 at roughly 14 a month. "
                 "August was the strongest at 22 and September the weakest at 7.",
}


def build_scripted_model() -> ScriptedModel:
    """A model that answers the demo questions without a network call."""
    return ScriptedModel(
        # Consulted when the agent asks for SQL.
        responses={
            "most revenue": _REVENUE,
            "best selling genres": _GENRES,
            "signed up each month": _SIGNUPS,
            "employee salaries": _SALARIES,
            "delete all cancelled": _DELETE,
        },
        # Consulted when the agent asks for an explanation; keyed on the
        # column header of the rendered result.
        answers={
            "country | revenue": _ANSWERS["revenue"],
            "genre | copies_sold": _ANSWERS["genre"],
            "month | new_customers": _ANSWERS["signed up"],
        },
        # Anything unrecognised: the model declines, as the system prompt
        # instructs when a question is outside the schema.
        default="CANNOT_ANSWER",
    )
