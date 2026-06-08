from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "solarshield.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL NOT NULL,
                explanation_short TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                analysis_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                txt_path TEXT,
                pdf_path TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rpa_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )


def create_analysis(result: dict[str, Any], source: str = "upload", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (created_at, source, risk_level, score, confidence, explanation_short, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                source,
                result["risk_level"],
                float(result["score"]),
                float(result["confidence"]),
                result["explanation_short"],
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return get_analysis(cursor.lastrowid)


def get_analysis(analysis_id: int) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        raise ValueError(f"Analysis {analysis_id} not found")
    return dict(row)


def list_analyses(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def create_report(analysis_id: int | None, title: str, content: str, txt_path: str | None, pdf_path: str | None) -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports (created_at, analysis_id, title, content, txt_path, pdf_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (utc_now(), analysis_id, title, content, txt_path, pdf_path),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def list_reports(limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def add_rpa_log(message: str, level: str = "INFO") -> dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO rpa_logs (created_at, level, message) VALUES (?, ?, ?)",
            (utc_now(), level, message),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM rpa_logs WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


def list_rpa_logs(limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM rpa_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

