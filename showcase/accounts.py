"""Local accounts, sign-in sessions, and question records."""

import hashlib
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from time import time


SESSION_SECONDS = 7 * 24 * 60 * 60
PAGE_SIZE = 25
USERNAME = re.compile(r"[A-Za-z0-9_]{3,24}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def password_digest(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)


def token_digest(token):
    return hashlib.sha256(token.encode("ascii")).digest()


class AccountStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    salt BLOB NOT NULL,
                    password_digest BLOB NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    digest BLOB PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    project TEXT NOT NULL,
                    question TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    route TEXT
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("PRAGMA foreign_keys = ON")
                yield db
        finally:
            db.close()

    def create_user(self, username, display_name, password, *, admin=False):
        if not isinstance(username, str) or not USERNAME.fullmatch(username.strip()):
            raise ValueError("账号需为 3 至 24 位英文字母、数字或下划线。")
        if not isinstance(display_name, str) or not 1 <= len(display_name.strip()) <= 30:
            raise ValueError("名称需为 1 至 30 个字符。")
        if not isinstance(password, str) or not 8 <= len(password) <= 128:
            raise ValueError("密码需为 8 至 128 个字符。")
        salt = secrets.token_bytes(16)
        digest = password_digest(password, salt)
        try:
            with self.connect() as db:
                cursor = db.execute(
                    """INSERT INTO users
                       (username, display_name, salt, password_digest, is_admin, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (username.strip(), display_name.strip(), salt, digest, admin, utc_now()),
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise ValueError("该账号已被使用，请登录或换一个账号。") from error
            raise

    def authenticate(self, username, password):
        if not isinstance(username, str) or not USERNAME.fullmatch(username.strip()):
            return None
        if not isinstance(password, str) or not 8 <= len(password) <= 128:
            return None
        with self.connect() as db:
            row = db.execute(
                "SELECT id, salt, password_digest, is_admin FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        digest = password_digest(password, row["salt"] if row else bytes(16))
        if row and secrets.compare_digest(digest, row["password_digest"]):
            return row
        return None

    def create_session(self, user_id):
        token = secrets.token_urlsafe(32)
        now = int(time())
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            db.execute(
                "INSERT INTO sessions (digest, user_id, expires_at) VALUES (?, ?, ?)",
                (token_digest(token), user_id, now + SESSION_SECONDS),
            )
        return token

    def viewer(self, token):
        if not token:
            return None
        with self.connect() as db:
            row = db.execute(
                """SELECT users.id, username, display_name, is_admin
                   FROM users JOIN sessions ON sessions.user_id = users.id
                   WHERE sessions.digest = ? AND sessions.expires_at > ?""",
                (token_digest(token), int(time())),
            ).fetchone()
        return dict(row) if row else None

    def revoke_session(self, token):
        if token:
            with self.connect() as db:
                db.execute("DELETE FROM sessions WHERE digest = ?", (token_digest(token),))

    def log_question(self, user_id, project, question):
        with self.connect() as db:
            cursor = db.execute(
                """INSERT INTO questions (user_id, project, question, created_at)
                   VALUES (?, ?, ?, ?)""",
                (user_id, project, question, utc_now()),
            )
            return cursor.lastrowid

    def finish_question(self, question_id, status, route=None):
        with self.connect() as db:
            db.execute(
                "UPDATE questions SET status = ?, route = ? WHERE id = ?",
                (status, route, question_id),
            )

    def overview(self, user_page, question_page):
        with self.connect() as db:
            db.execute("BEGIN")
            users = db.execute(
                """SELECT username, display_name, is_admin, created_at FROM users
                   ORDER BY id DESC LIMIT ? OFFSET ?""",
                (PAGE_SIZE, (user_page - 1) * PAGE_SIZE),
            ).fetchall()
            questions = db.execute(
                """SELECT users.username, users.display_name, questions.project, questions.question,
                          questions.created_at, questions.status, questions.route
                   FROM questions JOIN users ON users.id = questions.user_id
                   ORDER BY questions.id DESC LIMIT ? OFFSET ?""",
                (PAGE_SIZE, (question_page - 1) * PAGE_SIZE),
            ).fetchall()
            return {
                "user_count": db.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                "question_count": db.execute("SELECT COUNT(*) FROM questions").fetchone()[0],
                "users": [dict(row) for row in users],
                "questions": [dict(row) for row in questions],
                "page_size": PAGE_SIZE,
            }
