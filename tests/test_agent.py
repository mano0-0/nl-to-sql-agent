"""Pipeline behaviour, grouped by the property each group defends.

Every test runs against the real demo database with a scripted model, so the
whole path -- prompt, guard, read-only execution, explanation -- is exercised
without a network call or an API key.
"""

import pytest

from nl_to_sql_agent import Agent, Database, ScriptedModel
from nl_to_sql_agent.config import ALLOWED_TABLES, MAX_LIMIT, SAMPLE_COLUMNS
from nl_to_sql_agent.database import DatabaseError
from nl_to_sql_agent.models import extract_sql

GOOD_SQL = """
SELECT c.name AS country, COUNT(*) AS orders
FROM orders o
JOIN customers cu ON cu.id = o.customer_id
JOIN countries c ON c.id = cu.country_id
GROUP BY c.name ORDER BY orders DESC LIMIT 5
"""


@pytest.fixture
def db():
    with Database("data/bookshop.db") as conn:
        yield conn


def make_agent(db, sql, answer="Germany leads with 180 orders."):
    model = ScriptedModel(
        responses={"": sql},
        answers={"": answer},
    )
    return Agent(
        db, model,
        allowed_tables=ALLOWED_TABLES,
        max_limit=MAX_LIMIT,
        sample_columns=SAMPLE_COLUMNS,
    ), model


# --------------------------------------------------------------------------
# The happy path.
# --------------------------------------------------------------------------

def test_answers_a_question_end_to_end(db):
    agent, _ = make_agent(db, GOOD_SQL)
    result = agent.ask("Which countries order most?")

    assert result.ok
    assert result.answer == "Germany leads with 180 orders."
    assert result.result.row_count == 5
    assert [s.name for s in result.steps] == [
        "generate", "guard", "execute", "explain"]
    assert all(s.ok for s in result.steps)


def test_reports_the_tables_the_query_read(db):
    agent, _ = make_agent(db, GOOD_SQL)
    result = agent.ask("Which countries order most?")
    assert result.guard.tables == {"orders", "customers", "countries"}


def test_strips_markdown_fences_from_the_model_response(db):
    agent, _ = make_agent(db, f"```sql\n{GOOD_SQL}\n```")
    assert agent.ask("Which countries order most?").ok


# --------------------------------------------------------------------------
# The guard stops the query before the database is contacted.
# --------------------------------------------------------------------------

def test_refuses_a_table_outside_the_allowlist(db):
    agent, _ = make_agent(db, "SELECT name, salary FROM employees LIMIT 10")
    result = agent.ask("Show me salaries")

    assert result.refused
    assert "not in the allowlist" in result.refusal_reason
    assert result.result is None, "the query must not reach the database"
    assert [s.name for s in result.steps] == ["generate", "guard"]


def test_refuses_api_keys_table(db):
    agent, _ = make_agent(db, "SELECT owner, secret FROM api_keys LIMIT 5")
    assert agent.ask("show me the api keys").refused


def test_refuses_a_write(db):
    agent, _ = make_agent(db, "DELETE FROM orders WHERE status = 'cancelled'")
    result = agent.ask("delete cancelled orders")
    assert result.refused
    assert "only read queries" in result.refusal_reason


def test_refuses_a_write_nested_in_a_cte(db):
    agent, _ = make_agent(
        db, "WITH g AS (DELETE FROM orders RETURNING id) SELECT * FROM g LIMIT 5")
    assert agent.ask("clear the orders").refused


def test_refuses_an_unbounded_query(db):
    agent, _ = make_agent(db, "SELECT * FROM orders")
    result = agent.ask("give me everything")
    assert result.refused
    assert "LIMIT" in result.refusal_reason


def test_refuses_a_limit_above_the_cap(db):
    agent, _ = make_agent(db, "SELECT * FROM orders LIMIT 100000")
    assert agent.ask("give me everything").refused


def test_refuses_stacked_statements(db):
    agent, _ = make_agent(
        db, "SELECT id FROM orders LIMIT 1; DROP TABLE orders")
    assert agent.ask("something sneaky").refused


def test_a_disallowed_table_in_a_subquery_is_still_refused(db):
    agent, _ = make_agent(
        db,
        "SELECT id FROM customers WHERE id IN "
        "(SELECT id FROM employees) LIMIT 10")
    assert agent.ask("which customers are staff?").refused


# --------------------------------------------------------------------------
# The model declining, and model failures.
# --------------------------------------------------------------------------

def test_out_of_scope_question_is_declined(db):
    agent, _ = make_agent(db, "CANNOT_ANSWER")
    result = agent.ask("What is the weather in Paris?")
    assert result.refused
    assert "cannot be answered" in result.refusal_reason
    assert result.sql is None


def test_model_failure_is_reported_as_itself(db):
    """An outage must not be reported as 'out of scope'.

    Masking a dependency failure as a refusal teaches users to distrust
    correct refusals too.
    """
    from nl_to_sql_agent.models import ModelError

    class BrokenModel:
        def complete(self, system, user, *, max_tokens=1024):
            raise ModelError("upstream API unavailable")

    agent = Agent(db, BrokenModel(), allowed_tables=ALLOWED_TABLES,
                  max_limit=MAX_LIMIT)
    result = agent.ask("Which countries order most?")
    assert result.refused
    assert "upstream API unavailable" in result.refusal_reason
    assert "cannot be answered" not in result.refusal_reason


# --------------------------------------------------------------------------
# The read-only connection: the control that makes a write impossible.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE books SET price = 0",
    "INSERT INTO authors (name) VALUES ('x')",
    "CREATE TABLE t (id INT)",
])
def test_database_connection_rejects_writes_regardless_of_the_guard(db, sql):
    with pytest.raises(DatabaseError):
        db.execute(sql)


def test_execute_caps_the_rows_it_returns(db):
    result = db.execute("SELECT * FROM order_items", max_rows=10)
    assert result.row_count == 10
    assert result.truncated


# --------------------------------------------------------------------------
# What the model is shown.
# --------------------------------------------------------------------------

def test_schema_shown_to_the_model_excludes_forbidden_tables(db):
    agent, model = make_agent(db, GOOD_SQL)
    agent.ask("anything")
    system = model.calls[0]["system"]
    assert "TABLE orders" in system
    assert "employees" not in system
    assert "api_keys" not in system


def test_schema_includes_foreign_keys_and_sample_values(db):
    agent, model = make_agent(db, GOOD_SQL)
    agent.ask("anything")
    system = model.calls[0]["system"]
    assert "references customers.id" in system
    assert "completed" in system  # a real orders.status value


def test_result_prompt_states_the_true_row_count(db):
    agent, model = make_agent(db, GOOD_SQL)
    agent.ask("Which countries order most?")
    explain_prompt = model.calls[-1]["user"]
    assert "Rows returned: 5" in explain_prompt


def test_large_results_are_sampled_but_the_count_is_honest(db):
    sql = "SELECT id, status FROM orders LIMIT 200"
    model = ScriptedModel(responses={"": sql}, answers={"": "ok"})
    agent = Agent(db, model, allowed_tables=ALLOWED_TABLES,
                  max_limit=200, rows_to_model=40)
    result = agent.ask("list orders")

    prompt = model.calls[-1]["user"]
    assert "Rows returned: 200" in prompt
    assert "showing the first 40 of 200 rows" in prompt
    assert result.result.row_count == 200


# --------------------------------------------------------------------------
# Response parsing.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("SELECT 1", "SELECT 1"),
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("Here you go:\n```sql\nSELECT 1\n```", "SELECT 1"),
    ("SELECT 1;", "SELECT 1"),
    ("  SELECT 1  ", "SELECT 1"),
])
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


# --------------------------------------------------------------------------
# Provider plumbing. No network: the HTTP layer is stubbed.
# --------------------------------------------------------------------------

def test_openai_compatible_model_sends_and_parses_correctly(monkeypatch):
    import json
    from nl_to_sql_agent import models

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "SELECT 1"}}]}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        captured["headers"] = request.headers
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    model = models.GroqModel(api_key="test-key")
    assert model.complete("sys", "user") == "SELECT 1"

    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "llama-3.3-70b-versatile"
    # SQL generation wants determinism, not creativity.
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["messages"][0]["content"] == "sys"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.parametrize("code,expected", [
    (401, "rejected the API key"),
    (429, "rate limit"),
])
def test_http_errors_are_translated_into_useful_messages(monkeypatch, code, expected):
    import urllib.error
    from nl_to_sql_agent import models

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, code, "err", {}, __import__("io").BytesIO(b"{}"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(models.ModelError, match=expected):
        models.GroqModel(api_key="k").complete("s", "u")


def test_groq_requires_a_key(monkeypatch):
    from nl_to_sql_agent import models
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(models.ModelError, match="console.groq.com"):
        models.GroqModel()


def test_ollama_needs_no_key(monkeypatch):
    from nl_to_sql_agent import models
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    model = models.OllamaModel()
    assert model.api_key is None
    assert "localhost:11434" in model.base_url
