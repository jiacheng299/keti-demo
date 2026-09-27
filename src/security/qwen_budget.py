"""Cross-process concurrency and daily request budget for local and public servers."""
from contextlib import contextmanager
from datetime import date
from pathlib import Path
import sqlite3
import time
import uuid

from src.llm.qwen_client import QwenError

DEFAULT_DB = Path(__file__).resolve().parents[2] / "runs/settings/qwen_budget.sqlite"


class QwenBudget:
    def __init__(self, limit=40, path=None):
        self.limit = limit
        self.path = Path(path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS usage(day TEXT PRIMARY KEY, count INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS lease(id INTEGER PRIMARY KEY, owner TEXT, expires REAL)")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def used(self):
        with self.connection() as db:
            row = db.execute("SELECT count FROM usage WHERE day=?", (str(date.today()),)).fetchone()
        return row[0] if row else 0

    @contextmanager
    def session(self):
        owner = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT expires FROM lease WHERE id=1").fetchone()
            if row and row[0] > time.time():
                raise QwenError("已有 Qwen 检验正在运行，请完成后再试。")
            db.execute("INSERT OR REPLACE INTO lease VALUES (1,?,?)", (owner, time.time()+3600))
        try:
            yield self
        finally:
            with self.connection() as db:
                db.execute("DELETE FROM lease WHERE id=1 AND owner=?", (owner,))

    def reserve(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            day = str(date.today())
            row = db.execute("SELECT count FROM usage WHERE day=?", (day,)).fetchone()
            count = row[0] if row else 0
            if count >= self.limit:
                raise QwenError("今日 Qwen 请求额度已用完，请明天再试或在本机调整额度。")
            db.execute("INSERT OR REPLACE INTO usage VALUES (?,?)", (day, count+1))
