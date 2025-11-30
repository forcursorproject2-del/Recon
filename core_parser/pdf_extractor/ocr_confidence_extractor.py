"""
Модуль для извлечения текста с возвратом confidence score.

Расширяет функциональность PaddleOCR для возврата среднего confidence.
"""

import logging
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class OCRConfidenceExtractor:
    """Класс для извлечения текста с confidence score."""
    
    def __init__(self, ocr_extractor):
        """
        Инициализация.
        
        Args:
            ocr_extractor: Экземпляр PaddleOCRExtractor
        """
        self.ocr_extractor = ocr_extractor
    
    def extract_text_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Извлекает текст из изображения и возвращает средний confidence.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Tuple[текст, средний_confidence]: (str, float)
        """
        try:
            # Используем внутренний OCR объект напрямую
            result = self.ocr_extractor.ocr.ocr(image, cls=True)
            
            if not result or len(result) == 0 or result[0] is None:
                logger.warning("OCR не вернул результатов")
                return "", 0.0
            
            texts = []
            confidences = []
            
            # Обрабатываем результаты OCR
            for line_info in result[0]:
                try:
                    if line_info is None:
                        continue
                    
                    if not isinstance(line_info, (list, tuple)) or len(line_info) < 2:
                        continue
                    
                    text_data = line_info[1]
                    if text_data is None:
                        continue
                    
                    # Извлекаем текст и confidence
                    if isinstance(text_data, (list, tuple)):
                        text = text_data[0] if len(text_data) > 0 else ""
                        conf = float(text_data[1]) if len(text_data) > 1 else 0.0
                    else:
                        text = str(text_data)
                        conf = 1.0
                    
                    # Фильтруем по минимальному confidence
                    if conf > 0.1 and text.strip():
                        texts.append(text.strip())
                        confidences.append(conf)
                
                except (IndexError, TypeError, ValueError) as e:
                    logger.debug(f"Ошибка при обработке строки OCR: {e}")
                    continue
            
            if not texts:
                return "", 0.0
            
            # Объединяем текст
            full_text = "\n".join(texts)
            
            # Применяем постобработку для исправления ошибок OCR
            if full_text:
                try:
                    full_text = self.ocr_extractor.text_corrector.correct_text(full_text)
                    logger.debug("Применена постобработка текста OCR в confidence extractor")
                except Exception as e:
                    logger.warning(f"Ошибка при постобработке текста в confidence extractor: {e}")
            
            # Вычисляем средний confidence
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            logger.debug(f"Извлечено {len(texts)} строк, средний confidence: {avg_confidence:.3f}")
            
            return full_text, avg_confidence
        
        except Exception as e:
            logger.error(f"Ошибка при извлечении текста с confidence: {e}", exc_info=True)
            return "", 0.0

