import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class OcrCache:
    def __init__(self, cache_db: str = "ocr_cache.db"):
        self.cache_db = Path(cache_db)
        # Create a persistent connection for the object's lifetime
        self.conn = sqlite3.connect(str(self.cache_db), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_cache (
                page_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                lang TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self.conn.commit()

    def _get_page_hash(self, page_image_bytes: bytes, page_number: int) -> str:
        data = page_image_bytes + str(page_number).encode('utf-8')
        return hashlib.sha256(data).hexdigest()

    def get_cached_text(self, page_hash: str) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text FROM ocr_cache WHERE page_hash = ?",
            (page_hash,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def save_text(self, page_hash: str, text: str, lang: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO ocr_cache (page_hash, text, lang, timestamp) VALUES (?, ?, ?, ?)",
            (page_hash, text, lang, time.time())
        )
        self.conn.commit()

    def clear_old_cache(self, days: int = 30):
        cutoff = time.time() - (days * 24 * 60 * 60)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM ocr_cache WHERE timestamp < ?", (cutoff,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleared {deleted_count} old cache entries older than {days} days.")
        return deleted_count

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
