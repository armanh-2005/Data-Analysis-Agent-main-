import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

def connect(db_path: str) -> sqlite3.Connection:
    # تغییر ۱: افزایش تایم‌اوت برای جلوگیری از خطای قفل شدن
    conn = sqlite3.connect(db_path, timeout=60.0)
    
    # تغییر ۲: فعال‌سازی WAL Mode برای سرعت بالا در نوشتن
    conn.execute("PRAGMA journal_mode=WAL;")
    
    # تغییر ۳: تنظیم همگام‌سازی روی نرمال (امنیت خوب + سرعت بالا)
    conn.execute("PRAGMA synchronous=NORMAL;")
    
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