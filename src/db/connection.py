# connection.py
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional


def connect(db_path: str) -> sqlite3.Connection:
    # Enable foreign keys and return a connection with Row factory.
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session(db_path: str) -> Iterator[sqlite3.Connection]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = connect(db_path)
        yield conn
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()
