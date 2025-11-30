import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple
import logging

from .ocr_text_corrector import OCRTextCorrector
from .ocr_confidence_extractor import OCRConfidenceExtractor

logger = logging.getLogger(__name__)

class PaddleOCRExtractor:
    def __init__(self, cache_db="ocr_cache.db", model_dir=None):
        self.cache_db = cache_db
        if model_dir is None:
            model_dir = str(Path.home() / ".core_parser" / "paddleocr_models")
        self.model_dir = model_dir
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.lang = "ru"
        # Улучшенные настройки для сложных документов
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            det_model_dir=os.path.join(model_dir, "det"),
            rec_model_dir=os.path.join(model_dir, "rec", "ru"),
            cls_model_dir=os.path.join(model_dir, "cls"),
            # Более агрессивные параметры для плохих изображений
            det_db_thresh=0.2,  # Еще ниже порог для детекции текста
            det_db_box_thresh=0.4,  # Более низкий порог для боксов
            det_limit_side_len=3200,  # Увеличиваем для больших изображений
            rec_batch_num=8,  # Больше батч для лучшей обработки
            use_space_char=True,  # Использовать пробелы
            drop_score=0.2,  # Еще ниже порог отбрасывания результатов
            # Дополнительные параметры для улучшения
            det_db_unclip_ratio=1.8,  # Увеличиваем область детекции
            max_text_length=50,  # Увеличиваем максимальную длину текста
        )
        self.text_corrector = OCRTextCorrector()
        self.confidence_extractor = OCRConfidenceExtractor(self)
        self._init_cache()

    def _page_hash(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()

    def _init_cache(self):
        with sqlite3.connect(self.cache_db) as conn:
            # Создаем таблицу с полной структурой
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ocr_cache (
                    page_hash TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            # Проверяем структуру существующей таблицы и мигрируем при необходимости
            cursor = conn.execute("PRAGMA table_info(ocr_cache)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            # Добавляем недостающие колонки
            if 'lang' not in columns:
                try:
                    conn.execute("ALTER TABLE ocr_cache ADD COLUMN lang TEXT DEFAULT 'ru'")
                    conn.execute("UPDATE ocr_cache SET lang = 'ru' WHERE lang IS NULL")
                except sqlite3.OperationalError:
                    pass
            if 'timestamp' not in columns:
                try:
                    conn.execute("ALTER TABLE ocr_cache ADD COLUMN timestamp REAL")
                    conn.execute("UPDATE ocr_cache SET timestamp = ? WHERE timestamp IS NULL", (time.time(),))
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def _get_from_cache(self, page_hash: str) -> Optional[str]:
        with sqlite3.connect(self.cache_db) as conn:
            cur = conn.execute("SELECT text FROM ocr_cache WHERE page_hash=?", (page_hash,))
            row = cur.fetchone()
            return row[0] if row else None

    def _save_to_cache(self, page_hash: str, text: str):
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ocr_cache (page_hash, text, lang, timestamp) VALUES (?, ?, ?, ?)",
                (page_hash, text, self.lang, time.time())
            )
            conn.commit()

    def extract_text(self, image: np.ndarray) -> str:
        """
        Извлекает текст из изображения через OCR.
        
        Args:
            image: np.array в BGR формате
            
        Returns:
            Извлеченный текст
        """
        try:
            # Проверяем формат изображения и конвертируем при необходимости
            if len(image.shape) == 3:
                # Если изображение уже в BGR формате (3 канала)
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            elif len(image.shape) == 2:
                # Grayscale, конвертируем в BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                logger.warning(f"Неожиданный формат изображения: shape={image.shape}")
                return ""
            
            # Создаем хэш для кэширования
            _, buffer = cv2.imencode('.png', image)
            page_bytes = buffer.tobytes()
            page_hash = self._page_hash(page_bytes)

            # Проверяем кэш
            cached = self._get_from_cache(page_hash)
            if cached is not None:
                logger.debug("Текст найден в кэше")
                return cached

            # Выполняем OCR
            logger.debug("Выполняется OCR распознавание")
            result = self.ocr.ocr(image)
            
            if not result or len(result) == 0 or result[0] is None:
                logger.warning("OCR не вернул результатов")
                return ""
                
            lines = []
            confidence_scores = []
            
            # Обрабатываем результаты OCR
            # PaddleOCR возвращает: [[[bbox], (text, confidence)], ...]
            for line_info in result[0]:
                try:
                    if line_info is None:
                        continue
                    
                    # Проверяем структуру данных
                    if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
                        continue
                    
                    # line_info[1] должен быть (text, confidence) или [text, confidence]
                    text_data = line_info[1]
                    if text_data is None:
                        continue
                    
                    # Извлекаем текст и уверенность
                    if isinstance(text_data, (list, tuple)):
                        text = text_data[0] if len(text_data) > 0 else ""
                        conf = float(text_data[1]) if len(text_data) > 1 else 0.0
                    else:
                        # Если это просто строка
                        text = str(text_data)
                        conf = 1.0
                    
                    confidence_scores.append(conf)
                    
                    # Снижаем порог confidence для сложных документов
                    # Если документ не распознался с высоким порогом, пробуем с более низким
                    min_confidence = 0.4  # Снижен порог для плохих изображений
                    if conf >= min_confidence and text:
                        # Исправляем типичные ошибки PaddleOCR
                        text = text.strip()
                        if text:
                            lines.append(text)
                    elif conf >= 0.2 and text:  # Еще более низкий порог для очень плохих изображений
                        # Если нет строк с высоким confidence, берем строки с низким
                        # Это помогает для сложных документов (сканы, плохое качество)
                        text = text.strip()
                        if text and len(text) > 1:  # Минимальная длина текста снижена
                            lines.append(text)
                except (IndexError, TypeError, ValueError) as e:
                    logger.warning(f"Ошибка при обработке строки OCR: {e}, line_info: {line_info}")
                    continue

            if not lines:
                # Если нет строк с достаточной уверенностью, пробуем взять все строки с любой уверенностью
                logger.warning(f"OCR не распознал текст с порогом 0.4 (уверенность: {confidence_scores[:5] if confidence_scores else 'нет данных'})")
                # Пробуем с очень низким порогом для плохих изображений
                for line_info in result[0]:
                    try:
                        if line_info is None:
                            continue
                        if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
                            continue
                        text_data = line_info[1]
                        if text_data is None:
                            continue
                        if isinstance(text_data, (list, tuple)):
                            text = text_data[0] if len(text_data) > 0 else ""
                            conf = float(text_data[1]) if len(text_data) > 1 else 0.0
                        else:
                            text = str(text_data)
                            conf = 0.3
                        # Берем любой текст, даже с очень низким confidence
                        if text and len(text.strip()) > 0:
                            text = text.strip()
                            lines.append(text)
                    except (IndexError, TypeError, ValueError):
                        continue
                
                if not lines:
                    logger.warning("OCR не смог распознать текст даже с очень низким порогом confidence")
                    return ""

            full_text = "\n".join(lines)
            
            # Применяем постобработку для исправления ошибок OCR
            if full_text:
                try:
                    full_text = self.text_corrector.correct_text(full_text)
                    logger.debug("Применена постобработка текста OCR")
                except Exception as e:
                    logger.warning(f"Ошибка при постобработке текста: {e}")
            
            # Сохраняем в кэш (если не удалось - не критично, продолжаем работу)
            try:
                self._save_to_cache(page_hash, full_text)
            except Exception as cache_error:
                logger.warning(f"Не удалось сохранить в кэш: {cache_error}, но текст извлечен успешно")
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            logger.debug(f"OCR завершен: {len(lines)} строк, средняя уверенность: {avg_confidence:.2f}")
            
            return full_text
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}", exc_info=True)
            return ""
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Извлекает текст из изображения и возвращает средний confidence.
        
        Args:
            image: np.array в BGR формате
            
        Returns:
            Tuple[текст, средний_confidence]: (str, float)
        """
        return self.confidence_extractor.extract_text_with_confidence(image)
