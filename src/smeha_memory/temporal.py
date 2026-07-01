# SPDX-License-Identifier: Apache-2.0
"""Bitemporal fact-validity layer for agent memory.

A similarity/graph store answers "what is related to X?" but not "you said X, then Y — which is
CURRENT?" or "what was true at time T?". This sidecar adds bitemporal facts over SQLite:

  - add_fact auto-SUPERSEDES the prior (subject, predicate) fact (closes its valid_to + links supersedes),
  - current_fact -> the live one, as_of(t) -> the fact valid at time t, fact_history -> the chain,
  - open_facts -> everything still marked valid (a correct supersede leaves exactly ONE per (s,p)).

Non-invasive: its own `facts` table. Pass a sqlite3.Connection to share a DB, or a path / ":memory:".
"""
from __future__ import annotations

import sqlite3
import time


class TemporalFactStore:
    def __init__(self, conn_or_path=":memory:"):
        self.conn = (conn_or_path if isinstance(conn_or_path, sqlite3.Connection)
                     else sqlite3.connect(conn_or_path))
        self._init()

    def _init(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
            valid_from REAL NOT NULL, valid_to REAL,     -- NULL valid_to = still valid (open interval)
            supersedes INTEGER,                          -- id of the fact this one replaced
            confidence REAL DEFAULT 1.0, source TEXT, observed_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_f_sp ON facts(subject, predicate);
        """)
        self.conn.commit()

    _COLS = "id,subject,predicate,object,valid_from,valid_to,supersedes,confidence"

    def _row(self, r):
        if r is None:
            return None
        return dict(zip(self._COLS.split(","), r))

    def add_fact(self, subject, predicate, object, *, at=None, confidence=1.0, source=None):
        """Assert (s,p,o) valid from `at` (default now). Auto-supersedes the current (s,p) fact
        (closes its valid_to, links supersedes). Idempotent if the current fact already = object."""
        at = float(at if at is not None else time.time())
        cur = self.current_fact(subject, predicate)
        prev_id = None
        if cur is not None:
            if cur["object"] == object:
                return cur["id"]                          # same fact still holds -> no churn
            self.conn.execute("UPDATE facts SET valid_to=? WHERE id=? AND valid_to IS NULL",
                              (at, cur["id"]))
            prev_id = cur["id"]
        c = self.conn.execute(
            "INSERT INTO facts(subject,predicate,object,valid_from,valid_to,supersedes,"
            "confidence,source,observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (subject, predicate, object, at, None, prev_id, confidence, source, time.time()))
        self.conn.commit()
        return c.lastrowid

    def current_fact(self, subject, predicate):
        """The live (un-superseded) fact for (s,p), or None."""
        r = self.conn.execute(
            f"SELECT {self._COLS} FROM facts WHERE subject=? AND predicate=? AND valid_to IS NULL "
            "ORDER BY valid_from DESC LIMIT 1", (subject, predicate)).fetchone()
        return self._row(r)

    def as_of(self, subject, predicate, t):
        """The fact valid at time t (valid_from <= t < valid_to, or still open)."""
        r = self.conn.execute(
            f"SELECT {self._COLS} FROM facts WHERE subject=? AND predicate=? AND valid_from<=? "
            "AND (valid_to IS NULL OR valid_to>?) ORDER BY valid_from DESC LIMIT 1",
            (subject, predicate, t, t)).fetchone()
        return self._row(r)

    def fact_history(self, subject, predicate):
        return [self._row(r) for r in self.conn.execute(
            f"SELECT {self._COLS} FROM facts WHERE subject=? AND predicate=? ORDER BY valid_from",
            (subject, predicate)).fetchall()]

    def open_facts(self, subject, predicate):
        """All facts currently marked valid (valid_to IS NULL). A correct supersede leaves exactly
        ONE per (s,p); appending without closing the prior leaves several = ambiguous (a useful
        broken-control when testing)."""
        return [self._row(r) for r in self.conn.execute(
            f"SELECT {self._COLS} FROM facts WHERE subject=? AND predicate=? AND valid_to IS NULL",
            (subject, predicate)).fetchall()]
