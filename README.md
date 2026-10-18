# nl-to-sql-agent

Ask a database questions in plain English. The SQL the model writes is checked
before it runs.

```
$ nl-sql --offline "Which five countries generated the most revenue?"

  country  revenue
  ------------------
  Brazil   18,815.54
  France   18,216.55
  Spain    17,195.82
  Japan    17,048.49
  Germany  15,491.47

  Brazil led with €18,815.54, just ahead of France at €18,216.55.
```

```
$ nl-sql --offline "Show me the employee salaries"

  The generated query was rejected: table 'employees' is not in the allowlist
```

The second case is the interesting one. `employees` exists in the database, and
the model wrote a perfectly valid `SELECT` against it. The query never reached
the database.

## Try it

```bash
pip install -e ".[dev]"
python scripts/build_demo_db.py     # generates a small synthetic bookshop DB
nl-sql --offline --demo             # no API key, no network
nl-sql --offline --demo --show-sql  # with the pipeline trace
```

### Live answers, for free

`--offline` replays prepared answers, so only the demo questions work. To ask
anything you like, point it at a real model. **Groq's free tier needs no credit
card** and is the default:

```bash
# key from https://console.groq.com/keys -- email signup, no card
export GROQ_API_KEY=gsk_...
nl-sql "which authors sell best in Japan?"
nl-sql --show-sql "what is the average order value by region?"
```

Other providers, same pipeline:

```bash
nl-sql --provider ollama   "..."   # fully local, no key, no network
nl-sql --provider anthropic "..."  # needs ANTHROPIC_API_KEY
nl-sql --provider groq --model openai/gpt-oss-120b "..."   # larger
```

| provider | cost | setup |
|---|---|---|
| `groq` (default) | free tier, no card | `export GROQ_API_KEY=...` |

Groq's default here is `openai/gpt-oss-20b`, which a free key can use. Its
Llama models are Enterprise-tier and return `404 model_not_found` on a free
key. To see what yours can reach:

```bash
curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

| `ollama` | free, runs locally | `brew install ollama && ollama pull qwen2.5-coder:7b` |
| `anthropic` | paid | `pip install '.[anthropic]'`, `ANTHROPIC_API_KEY` |

Only the model call differs. Guard, database and prompts are identical, which
is the point of keeping the model behind a two-method interface.

## The pipeline

```
  "Which five countries generated the most revenue?"
        │
   1 ── generate   ask the model for SQL, given only the allowed schema
        │
   2 ── guard      validate it with llm-sql-guard  ──✗──> refuse, with a reason
        │
   3 ── execute    run it on a read-only connection
        │
   4 ── explain    ask the model to describe the rows it got back
        │
  "Brazil led with €18,815.54..."
```

`--show-sql` prints this trace with timings, which is also how you debug a
surprising answer: you can see exactly what was generated and what it read.

## Two controls, doing different jobs

The database connection is opened read-only (`mode=ro`), and the generated SQL
is separately validated. These overlap, and the overlap is deliberate — but
they are not interchangeable.

| | read-only connection | llm-sql-guard |
|---|---|---|
| `DELETE`, `DROP`, `UPDATE` | **blocked by the engine** | blocked (redundantly) |
| reading `employees` | allowed — it's a read | **blocked** |
| `SELECT *` with no `LIMIT` | allowed | **blocked** |
| `pg_read_file()`, `dblink()` | allowed — they're reads | **blocked** |
| explains itself to the user | no | **yes** |
| survives a bug in this code | **yes** | no |

**The connection is the real guarantee.** It is enforced by the database, so a
bug in this program cannot produce a write. If you keep one control, keep that.

**The guard covers what read-only does not.** A read-only role happily runs
`SELECT * FROM salaries`, returns fifty million rows, or reads a server file
through `pg_read_file()`. Those are all reads.

Being specific about the redundancy: **the write check is the one part of the
guard that read-only makes unnecessary for safety.** It is kept for two lesser
reasons — the user gets `only read queries are allowed, found DELETE` instead
of `attempt to write a readonly database`, and a blocked write is a clean
signal worth logging, since it means something upstream went wrong. The checks
that genuinely add protection are the table allowlist, the row cap, the
function denylist, and single-statement enforcement.

## What the agent may not touch

The demo database contains two tables the agent is never allowed to read:

```
employees (id, name, salary, national_id)
api_keys  (id, owner, secret)
```

They are absent from `ALLOWED_TABLES` in `config.py`, so:

1. they never appear in the schema the model is shown, and
2. if the model names one anyway, the guard refuses the query.

Step 2 matters because step 1 is not a control. A model can guess a table name,
and a user can ask leadingly. Not showing it is a nudge; the allowlist is the
boundary.

## Design notes

**The guard is a step, not an instruction.** Telling a model "only write safe
SQL" in a prompt is a request it may decline. Validating the output is a rule
it cannot. Prompt text is good for quality; it is not an access control.

**A model failure is not a refusal.** If the API is unreachable, the user is
told that — not "your question is out of scope". Masking an outage as a refusal
trains people to distrust the correct refusals too.

**Results are sampled, but the count is honest.** Only the first 40 rows go to
the model, along with the true row count and a note saying so. Sending a sample
without the count produces confident summaries of 40 rows as though they were
all of them.

**Runaway queries are interrupted.** SQLite's progress handler aborts a
statement that burns through too many VM steps, so a cartesian product fails
fast rather than hanging the process.

**The model is a protocol, not a dependency.** `LanguageModel` is a two-method
interface. The full pipeline is tested against a scripted implementation, which
is why the 41 tests need no API key and run in under a tenth of a second. Groq,
Ollama and anything else speaking the OpenAI chat shape share one 60-line
implementation; Anthropic has its own because its API differs.

## Project layout

```
src/nl_to_sql_agent/
    agent.py      the four-step pipeline
    database.py   read-only connection, schema rendering
    models.py     the LanguageModel protocol; Groq, Ollama, Claude, scripted
    config.py     the table allowlist and limits
    cli.py        terminal interface
    demo.py       scripted responses for the offline demo
scripts/
    build_demo_db.py   generates the synthetic database (fixed seed)
tests/                 41 tests, no network required
```

## Related

[llm-sql-guard](https://github.com/mano0-0/llm-sql-guard) — the validation
library used in step 2, which walks the parsed syntax tree rather than
pattern-matching the query text.

## Licence

MIT
