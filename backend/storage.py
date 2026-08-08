from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "syncspace.db"


def _connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    with _connection() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
              content TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE TABLE IF NOT EXISTS mcp_servers (
              id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, command TEXT NOT NULL,
              args_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            """
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def create_session(title: str = "New conversation") -> dict:
    item = {"id": str(uuid.uuid4()), "title": title, "created_at": _now()}
    with _connection() as db:
        db.execute("INSERT INTO sessions VALUES (:id, :title, :created_at)", item)
    return item


def list_sessions() -> list[dict]:
    with _connection() as db:
        rows = db.execute(
            """
            SELECT s.id, s.title, s.created_at,
                   (SELECT m.content
                    FROM messages AS m
                    WHERE m.session_id = s.id AND m.role = 'user'
                    ORDER BY m.created_at
                    LIMIT 1) AS first_message
            FROM sessions AS s
            ORDER BY s.created_at DESC
            """
        )
        sessions = [dict(row) for row in rows]

    for session in sessions:
        first_message = session.pop("first_message")
        if session["title"] == "New conversation" and first_message:
            session["title"] = " ".join(first_message.split())[:60]
    return sessions


def get_messages(session_id: str) -> list[dict]:
    with _connection() as db:
        return [dict(row) for row in db.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at", (session_id,)
        )]


def add_message(session_id: str, role: str, content: str) -> None:
    with _connection() as db:
        db.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, _now()),
        )


def save_mcp_server(name: str, command: str, args: list[str]) -> dict:
    item = {"id": str(uuid.uuid4()), "name": name, "command": command, "args_json": json.dumps(args), "created_at": _now()}
    with _connection() as db:
        db.execute(
            "INSERT INTO mcp_servers VALUES (:id, :name, :command, :args_json, :created_at) "
            "ON CONFLICT(name) DO UPDATE SET command=excluded.command, args_json=excluded.args_json",
            item,
        )
    return {**item, "args": args}


def list_mcp_servers() -> list[dict]:
    with _connection() as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM mcp_servers ORDER BY name")]
    return [{**row, "args": json.loads(row.pop("args_json"))} for row in rows]
