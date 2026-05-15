import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATABASE_PATH, LOGS_DIR


class InspectionDatabase:
    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                defects INTEGER,
                severity TEXT,
                risk TEXT,
                recommendation TEXT,
                area REAL,
                spread REAL,
                occupancy REAL,
                confidence REAL,
                report_path TEXT
            )
            """
        )
        self.connection.commit()

    def log_inspection(self, entry: dict[str, Any]) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO inspections (
                timestamp, image_path, defects, severity, risk,
                recommendation, area, spread, occupancy, confidence, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.get("timestamp", datetime.utcnow().isoformat()),
                entry.get("image_path", ""),
                entry.get("defects", 0),
                entry.get("severity", "Unknown"),
                entry.get("risk", "Unknown"),
                entry.get("recommendation", "Monitor"),
                entry.get("area", 0.0),
                entry.get("spread", 0.0),
                entry.get("occupancy", 0.0),
                entry.get("confidence", 0.0),
                entry.get("report_path", ""),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def fetch_recent_inspections(self, limit: int = 50) -> list[sqlite3.Row]:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM inspections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()

    def fetch_all_inspections(self) -> list[sqlite3.Row]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM inspections ORDER BY id DESC")
        return cursor.fetchall()

    def close(self) -> None:
        self.connection.close()
