"""SQLite-backed conversation memory.

Follow-up questions ("what about its side effects?") are unanswerable without
the previous turn, so each completed run is persisted and replayed to the
planner and synthesizer as context.

Only the question, the summary, and the PMIDs are stored - not full abstracts.
Context should orient the next turn, not refill its window.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    run_id       TEXT NOT NULL,
    question     TEXT NOT NULL,
    summary      TEXT NOT NULL,
    pmids        TEXT NOT NULL DEFAULT '[]',
    grade        TEXT NOT NULL DEFAULT '',
    answered     INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);
"""


@dataclass(slots=True)
class Turn:
    """One completed question/answer exchange."""

    session_id: str
    run_id: str
    question: str
    summary: str
    pmids: list[str]
    grade: str
    answered: bool
    created_at: str


class ConversationStore:
    """Persists and retrieves conversation turns."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    def add_turn(
        self,
        session_id: str,
        run_id: str,
        question: str,
        summary: str,
        pmids: list[str],
        grade: str = "",
        answered: bool = True,
    ) -> None:
        """Record a completed exchange."""
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO turns (session_id, run_id, question, summary, pmids, grade, "
                "answered, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    run_id,
                    question,
                    summary,
                    json.dumps(pmids),
                    grade,
                    int(answered),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            conn.commit()

    def history(self, session_id: str, limit: int = 5) -> list[Turn]:
        """Return the most recent turns, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            Turn(
                session_id=row["session_id"],
                run_id=row["run_id"],
                question=row["question"],
                summary=row["summary"],
                pmids=json.loads(row["pmids"]),
                grade=row["grade"],
                answered=bool(row["answered"]),
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]

    def context_for(self, session_id: str, limit: int = 3, max_chars: int = 2000) -> str:
        """Render recent history as prompt context.

        Bounded in both turns and characters so a long session cannot crowd out
        the retrieved evidence in the synthesis window.
        """
        turns = self.history(session_id, limit=limit)
        if not turns:
            return ""

        blocks: list[str] = []
        for index, turn in enumerate(turns, start=1):
            summary = turn.summary if len(turn.summary) <= 400 else turn.summary[:400] + "…"
            citations = ", ".join(turn.pmids[:6])
            block = f"Turn {index}\nQ: {turn.question}\nA: {summary}"
            if citations:
                block += f"\nCited PMIDs: {citations}"
            blocks.append(block)

        context = "\n\n".join(blocks)
        if len(context) > max_chars:
            context = context[-max_chars:]
            context = context[context.find("\n") + 1 :]
        return context

    def sessions(self) -> list[str]:
        """All known session ids, most recently used first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT session_id, MAX(id) AS last FROM turns "
                "GROUP BY session_id ORDER BY last DESC"
            ).fetchall()
        return [row["session_id"] for row in rows]

    def clear(self, session_id: str) -> int:
        """Delete a session's history; returns the number of turns removed."""
        with closing(self._connect()) as conn:
            cursor = conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount
