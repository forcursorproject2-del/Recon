import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import hashlib
import sqlite3
from pathlib import Path

class PaddleOCRExtractor:
    def __init__(self, cache_db="ocr_cache.db", model_dir=None):
        self.cache_db = cache_db
        if model_dir is None:
            model_dir = str(Path.home() / ".core_parser" / "paddleocr_models")
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ru",
            use_gpu=False,
            show_log=False,
            det=True,
            rec=True,
            cls=True,
            det_model_dir=os.path.join(model_dir, "det"),
            rec_model_dir=os.path.join(model_dir, "rec", "ru"),
            cls_model_dir=os.path.join(model_dir, "cls"),
        )
        self._init_cache()

    def _page_hash(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    def _init_cache(self):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    page_hash TEXT PRIMARY KEY,
                    text TEXT
                )
            """)

    def _get_from_cache(self, page_hash: str) -> str | None:
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.execute("SELECT text FROM ocr_cache WHERE page_hash=?", (page_hash,))
            row = cur.fetchone()
            return row[0] if row else None

    def _save_to_cache(self, page_hash: str, text: str):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("INSERT OR REPLACE INTO ocr_cache (page_hash, text) VALUES (?, ?)",
                        (page_hash, text))

    def extract_text(self, image: np.ndarray) -> str:
        # image — np.array в BGR от PyMuPDF
        _, buffer = cv2.imencode('.png', image)
        page_bytes = buffer.tobytes()
        page_hash = self._page_hash(page_bytes)

        cached = self._get_from_cache(page_hash)
        if cached is not None:
            return cached

        result = self.ocr.ocr(image, det=True, rec=True, cls=True, merge_text=True)
        lines = []
        for line_info in result[0]:
            text = line_info[1][0]
            conf = line_info[1][1]
            if conf >= 0.55:
                # Исправляем типичные ошибки PaddleOCR
                text = text.replace('ё', 'ё').replace('Ё', 'Ё')  # уже нормально, но на всякий
                lines.append(text)

        full_text = "\n".join(lines)
        self._save_to_cache(page_hash, full_text)
        return full_text
