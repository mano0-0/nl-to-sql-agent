#!/usr/bin/env python3
"""Generate the demo SQLite database.

A small fictional online bookshop. Entirely synthetic -- generated from a fixed
seed so every clone of this repository produces a byte-identical database, and
no external download is required.

The schema deliberately includes two tables the agent is NOT allowed to read
(`employees` and `api_keys`), so the access controls have something real to
protect. See README: "What the agent may not touch".
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bookshop.db"
SEED = 20261018

SCHEMA = """
CREATE TABLE countries (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    region      TEXT NOT NULL
);

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    country_id  INTEGER NOT NULL REFERENCES countries(id),
    signed_up   DATE NOT NULL
);

CREATE TABLE authors (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE books (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    author_id   INTEGER NOT NULL REFERENCES authors(id),
    genre       TEXT NOT NULL,
    price       REAL NOT NULL,
    published   INTEGER NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    ordered_at  DATE NOT NULL,
    status      TEXT NOT NULL
);

CREATE TABLE order_items (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    book_id     INTEGER NOT NULL REFERENCES books(id),
    quantity    INTEGER NOT NULL,
    unit_price  REAL NOT NULL
);

-- Deliberately off-limits to the agent.
CREATE TABLE employees (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    salary      REAL NOT NULL,
    national_id TEXT NOT NULL
);

CREATE TABLE api_keys (
    id          INTEGER PRIMARY KEY,
    owner       TEXT NOT NULL,
    secret      TEXT NOT NULL
);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date     ON orders(ordered_at);
CREATE INDEX idx_items_order     ON order_items(order_id);
CREATE INDEX idx_items_book      ON order_items(book_id);
CREATE INDEX idx_customers_ctry  ON customers(country_id);
"""

COUNTRIES = [
    ("Germany", "Europe"), ("France", "Europe"), ("Spain", "Europe"),
    ("Italy", "Europe"), ("Netherlands", "Europe"), ("Poland", "Europe"),
    ("United States", "North America"), ("Canada", "North America"),
    ("Brazil", "South America"), ("Japan", "Asia"), ("India", "Asia"),
    ("Australia", "Oceania"),
]

GENRES = ["Fiction", "Science Fiction", "History", "Biography",
          "Poetry", "Technology", "Cookery", "Travel"]

FIRST = ["Anna", "Luca", "Mira", "Tomas", "Sofia", "Jonas", "Elena", "Omar",
         "Nina", "Paul", "Yuki", "Ravi", "Clara", "Hugo", "Lena", "Mateo",
         "Ida", "Noah", "Zara", "Felix"]
LAST = ["Weber", "Rossi", "Dubois", "Nowak", "Silva", "Tanaka", "Patel",
        "Meyer", "Garcia", "Jensen", "Kovac", "Haddad", "Lindqvist", "Moreau"]

TITLE_A = ["The Silent", "A Brief", "Letters from", "The Last", "Notes on",
           "Winter in", "The Art of", "Shadows of", "Return to", "The Quiet"]
TITLE_B = ["Harbour", "History", "Vienna", "Cartographer", "Small Things",
           "Lisbon", "Listening", "the Steppe", "Mornings", "Machines",
           "the Archive", "Salt"]


def main() -> None:
    rng = random.Random(SEED)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)

    db.executemany("INSERT INTO countries (name, region) VALUES (?, ?)", COUNTRIES)
    n_countries = len(COUNTRIES)

    authors = [(f"{rng.choice(FIRST)} {rng.choice(LAST)}",) for _ in range(40)]
    db.executemany("INSERT INTO authors (name) VALUES (?)", authors)

    books = []
    for _ in range(120):
        title = f"{rng.choice(TITLE_A)} {rng.choice(TITLE_B)}"
        books.append((
            title,
            rng.randint(1, len(authors)),
            rng.choice(GENRES),
            round(rng.uniform(6.0, 48.0), 2),
            rng.randint(1985, 2026),
        ))
    db.executemany(
        "INSERT INTO books (title, author_id, genre, price, published) "
        "VALUES (?, ?, ?, ?, ?)", books)

    start = date(2024, 1, 1)
    customers = []
    for i in range(300):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        email = f"{name.lower().replace(' ', '.')}{i}@example.com"
        customers.append((
            name, email,
            rng.randint(1, n_countries),
            (start + timedelta(days=rng.randint(0, 700))).isoformat(),
        ))
    db.executemany(
        "INSERT INTO customers (name, email, country_id, signed_up) "
        "VALUES (?, ?, ?, ?)", customers)

    orders, items = [], []
    order_id = 0
    for _ in range(1800):
        order_id += 1
        cust = rng.randint(1, len(customers))
        when = start + timedelta(days=rng.randint(0, 720))
        status = rng.choices(
            ["completed", "shipped", "cancelled", "refunded"],
            weights=[72, 18, 6, 4])[0]
        orders.append((cust, when.isoformat(), status))
        for _ in range(rng.randint(1, 4)):
            book = rng.randint(1, len(books))
            items.append((order_id, book, rng.randint(1, 3), books[book - 1][3]))

    db.executemany(
        "INSERT INTO orders (customer_id, ordered_at, status) VALUES (?, ?, ?)",
        orders)
    db.executemany(
        "INSERT INTO order_items (order_id, book_id, quantity, unit_price) "
        "VALUES (?, ?, ?, ?)", items)

    # Off-limits tables: plausible-looking, so a successful read would clearly
    # be a failure of the access controls rather than an empty-table non-event.
    db.executemany(
        "INSERT INTO employees (name, salary, national_id) VALUES (?, ?, ?)",
        [(f"{rng.choice(FIRST)} {rng.choice(LAST)}",
          round(rng.uniform(38000, 120000), 2),
          f"ID-{rng.randint(10**7, 10**8 - 1)}") for _ in range(25)])
    db.executemany(
        "INSERT INTO api_keys (owner, secret) VALUES (?, ?)",
        [(f"service-{i}", f"sk_live_{rng.getrandbits(80):020x}") for i in range(8)])

    db.commit()

    counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("countries", "authors", "books", "customers",
                        "orders", "order_items", "employees", "api_keys")}
    db.close()

    print(f"wrote {DB_PATH}")
    for t, n in counts.items():
        print(f"  {t:12} {n:>6}")


if __name__ == "__main__":
    main()
