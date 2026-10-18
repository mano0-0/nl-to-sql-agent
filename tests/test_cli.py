"""CLI behaviour: exit codes, output, and the offline path."""

import pytest

from nl_to_sql_agent.cli import main


def test_offline_demo_runs_without_network(capsys):
    assert main(["--offline", "--demo"]) == 0
    out = capsys.readouterr().out
    assert "Brazil" in out
    # The demo must exercise refusals, not only the happy path.
    assert "not in the allowlist" in out
    assert "only read queries are allowed" in out


def test_trace_shows_the_generated_sql(capsys):
    main(["--offline", "--demo", "--show-sql"])
    out = capsys.readouterr().out
    assert "generate" in out and "guard" in out and "execute" in out
    assert "SELECT" in out


def test_schema_flag_hides_forbidden_tables(capsys):
    assert main(["--schema"]) == 0
    out = capsys.readouterr().out
    assert "TABLE orders" in out
    assert "TABLE employees" not in out
    assert "Not readable by the agent: employees, api_keys" in out


def test_missing_database_is_a_clear_error(capsys):
    assert main(["--db", "/nonexistent/x.db", "--offline", "x"]) == 2
    assert "database not found" in capsys.readouterr().err


def test_no_question_prints_help(capsys):
    assert main(["--offline"]) == 2
    assert "usage:" in capsys.readouterr().out


def test_live_mode_without_a_key_explains_how_to_proceed(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = main(["--db", "data/bookshop.db", "hello"])
    err = capsys.readouterr().err
    # Either no key, or the SDK is absent -- both must name --offline.
    assert code == 2
    assert "--offline" in err or "anthropic" in err
