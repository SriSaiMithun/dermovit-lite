"""
Authentication for DermoViT-Lite: signup, login, and JWT verification.

Design choices, and why:
- SQLite (stdlib, no extra service) for user storage. Render's free-tier
  disk persists across spin-down/wake cycles (only wiped on a new
  deploy), which is fine for a student project - a managed database
  would be the real-world upgrade, noted as a limitation in the report.
- Passwords hashed with werkzeug's generate_password_hash (PBKDF2) -
  never stored or logged in plaintext.
- JWT (PyJWT) for stateless auth: /predict requires a valid
  'Authorization: Bearer <token>' header. Tokens expire after 24h.
"""

import datetime
import os
import sqlite3

import jwt
from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "users.db")
# In production this MUST come from an environment variable, never a
# hardcoded default - see README for setting JWT_SECRET_KEY on Render.
JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message)."""
    if not username or not password:
        return False, "Username and password are required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."

    conn = sqlite3.connect(DB_PATH)
    try:
        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    finally:
        conn.close()


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """Returns (success, message_or_token)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or not check_password_hash(row[0], password):
        return False, "Invalid username or password."

    token = jwt.encode(
        {
            "username": username,
            "exp": datetime.datetime.utcnow()
            + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
        },
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    return True, token


def verify_token(token: str) -> tuple[bool, str]:
    """Returns (valid, username_or_error_message)."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return True, payload["username"]
    except jwt.ExpiredSignatureError:
        return False, "Token expired, please log in again."
    except jwt.InvalidTokenError:
        return False, "Invalid token."


init_db()
