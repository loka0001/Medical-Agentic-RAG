import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class Storage:
    def __init__(self, db_path="data/app.sqlite3"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._init_db()
        except sqlite3.OperationalError:
            self.db_path = self.db_path.with_name("app_live.sqlite3")
            if self.db_path.exists() and self.db_path.stat().st_size == 0:
                self.db_path.unlink()
            self._init_db()

    def save_case(self, patient_id, payload):
        self._execute(
            "INSERT INTO cases (patient_id, payload, created_at) VALUES (?, ?, ?)",
            (patient_id, self._dump_json(payload), self._now()),
        )

    def save_decision(self, patient_id, status, decision):
        self._execute(
            "INSERT INTO decisions (patient_id, status, decision, created_at) VALUES (?, ?, ?, ?)",
            (patient_id, status, self._dump_json(decision), self._now()),
        )

    def save_feedback(self, patient_id, feedback_type, payload):
        self._execute(
            "INSERT INTO feedback (patient_id, feedback_type, payload, created_at) VALUES (?, ?, ?, ?)",
            (patient_id, feedback_type, self._dump_json(payload), self._now()),
        )

    def save_document(self, source, path=None, metadata=None):
        self._execute(
            "INSERT INTO documents (source, path, metadata, created_at) VALUES (?, ?, ?, ?)",
            (source, path, self._dump_json(metadata or {}), self._now()),
        )

    def get_patient_history(self, patient_id, limit=20):
        return {
            "patient_id": patient_id,
            "cases": self._fetch_payloads(
                "SELECT payload, created_at FROM cases WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
                (patient_id, limit),
            ),
            "decisions": self._fetch_rows(
                "SELECT status, decision, created_at FROM decisions WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
                (patient_id, limit),
                json_fields=("decision",),
            ),
            "feedback": self._fetch_rows(
                "SELECT feedback_type, payload, created_at FROM feedback WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
                (patient_id, limit),
                json_fields=("payload",),
            ),
        }

    def list_documents(self):
        rows = self._query(
            "SELECT source, path, metadata, created_at FROM documents ORDER BY id DESC"
        )
        return [
            {
                "source": row["source"],
                "path": row["path"],
                "metadata": self._load_json(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def state_summary(self):
        return {
            "cases": self._count("cases"),
            "decisions": self._count("decisions"),
            "feedback": self._count("feedback"),
            "documents": self._count("documents"),
        }

    def _init_db(self):
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                status TEXT NOT NULL,
                decision TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                path TEXT,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _execute(self, sql, params=()):
        with self._connect() as connection:
            connection.execute(sql, params)
            connection.commit()

    def _query(self, sql, params=()):
        with self._connect() as connection:
            return connection.execute(sql, params).fetchall()

    def _fetch_payloads(self, sql, params):
        return [
            {"payload": self._load_json(row["payload"]), "created_at": row["created_at"]}
            for row in self._query(sql, params)
        ]

    def _fetch_rows(self, sql, params, json_fields=()):
        rows = []
        for row in self._query(sql, params):
            item = dict(row)
            for field in json_fields:
                item[field] = self._load_json(item[field])
            rows.append(item)
        return rows

    def _count(self, table):
        return self._query(f"SELECT COUNT(*) AS total FROM {table}")[0]["total"]

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _dump_json(self, value):
        return json.dumps(self._json_safe(value), ensure_ascii=False)

    def _load_json(self, value):
        return json.loads(value) if value else None

    def _json_safe(self, value):
        if isinstance(value, BaseModel):
            return self._json_safe(value.model_dump())
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: self._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._json_safe(item) for item in value]
        return value
