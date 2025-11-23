import pytest
import tempfile
import os
from pathlib import Path
from core_parser.pdf_extractor.ocr_cache import OcrCache
import time

class TestOcrCache:
    def setup_method(self):
        self.cache = OcrCache(":memory:")  # Use in-memory DB for tests

    def test_save_and_get_text(self):
        page_hash = "test_hash_123"
        text = "Sample OCR text"
        lang = "ru"

        self.cache.save_text(page_hash, text, lang)
        retrieved = self.cache.get_cached_text(page_hash)

        assert retrieved == text

    def test_get_nonexistent_text(self):
        retrieved = self.cache.get_cached_text("nonexistent_hash")
        assert retrieved is None

    def test_clear_old_cache(self):
        # Save some old entries
        old_hash = "old_hash"
        self.cache.save_text(old_hash, "old text", "ru")
        # Manually set old timestamp
        cursor = self.cache.conn.cursor()
        cursor.execute("UPDATE ocr_cache SET timestamp = ? WHERE page_hash = ?", (time.time() - 60*60*24*40, old_hash))
        self.cache.conn.commit()

        # Save new entry
        new_hash = "new_hash"
        self.cache.save_text(new_hash, "new text", "ru")

        # Clear old cache
        deleted = self.cache.clear_old_cache(days=30)

        assert deleted == 1
        assert self.cache.get_cached_text(old_hash) is None
        assert self.cache.get_cached_text(new_hash) == "new text"

    def test_page_hash_generation(self):
        img_bytes = b"fake_image_data"
        page_num = 1
        hash1 = self.cache._get_page_hash(img_bytes, page_num)
        hash2 = self.cache._get_page_hash(img_bytes, page_num)
        assert hash1 == hash2
        # Different data should give different hash
        hash3 = self.cache._get_page_hash(b"different_data", page_num)
        assert hash1 != hash3
