"""
Специализированные варианты предобработки для актов сверки.

Содержит агрессивные методы, специально разработанные для обработки
плохих сканов актов сверки с печатями, серым фоном и слабыми линиями таблиц.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

from .base_preprocessor import BasePreprocessor
from .seal_remover import SealRemover

logger = logging.getLogger(__name__)


class ReconciliationPreprocessor(BasePreprocessor):
    """Специализированные варианты для актов сверки."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.seal_remover = SealRemover(config=self.config)
    
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает специализированные варианты для актов сверки.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Список кортежей (название, обработанное_изображение)
        """
        variants = []
        
        # Вариант 1: CLAHE + морфология + удаление синих печатей
        try:
            img1 = self._reconciliation_aggressive_v1(image)
            variants.append(('reconciliation_aggressive_v1', img1))
        except Exception as e:
            logger.warning(f"Ошибка reconciliation_aggressive_v1: {e}")
        
        # Вариант 2: Grayscale + очень сильный adaptiveThreshold + dilation линий
        try:
            img2 = self._reconciliation_aggressive_v2(image)
            variants.append(('reconciliation_aggressive_v2', img2))
        except Exception as e:
            logger.warning(f"Ошибка reconciliation_aggressive_v2: {e}")
        
        return variants
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Базовый метод - использует первый агрессивный вариант."""
        return self._reconciliation_aggressive_v1(image)
    
    def _reconciliation_aggressive_v1(self, image: np.ndarray) -> np.ndarray:
        """
        Вариант 1: CLAHE + морфология + удаление синих печатей.
        
        Специально для серых документов с синими печатями.
        """
        # Масштабируем для лучшего качества
        scaled = self._scale_image_for_ocr(image, scale_factor=3.5)
        
        # 1. Удаление синих печатей (более агрессивно)
        img = self.seal_remover.remove_blue_seals(scaled)
        img = self.seal_remover.remove_purple_seals(img)  # Также удаляем фиолетовые
        
        # 2. CLAHE очень агрессивный
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Двойной CLAHE для очень серых документов
        clahe1 = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        enhanced = clahe1.apply(gray)
        clahe2 = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        enhanced = clahe2.apply(enhanced)
        
        # 3. Морфология для усиления текста (более агрессивно)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # 4. Усиление контраста (более агрессивно)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=15)
        
        # 5. Unsharp mask для резкости
        gaussian = cv2.GaussianBlur(enhanced, (5, 5), 0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _reconciliation_aggressive_v2(self, image: np.ndarray) -> np.ndarray:
        """
        Вариант 2: Grayscale + очень сильный adaptiveThreshold + dilation линий таблицы.
        
        Специально для документов с очень слабыми линиями таблиц.
        """
        # Масштабируем для лучшего качества
        scaled = self._scale_image_for_ocr(image, scale_factor=3.5)
        
        # 1. Удаление синих и фиолетовых печатей перед обработкой
        scaled = self.seal_remover.remove_blue_seals(scaled)
        scaled = self.seal_remover.remove_purple_seals(scaled)
        
        # 2. Grayscale
        if len(scaled.shape) == 3:
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = scaled.copy()
        
        # 3. Улучшение контраста перед бинаризацией (более агрессивно)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 4. Дополнительное растяжение гистограммы
        gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
        
        # 5. Очень сильный adaptiveThreshold (увеличиваем блок)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            61, 18  # Еще больший блок, еще больший C
        )
        
        # 6. Dilation линий таблицы (вертикальные и горизонтальные) - более агрессивно
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        
        vertical = cv2.dilate(binary, kernel_v, iterations=2)
        horizontal = cv2.dilate(binary, kernel_h, iterations=2)
        table_lines = cv2.bitwise_or(vertical, horizontal)
        
        # 7. Комбинируем бинаризацию с линиями таблицы
        result = cv2.bitwise_and(binary, table_lines)
        result = cv2.addWeighted(binary, 0.75, result, 0.25, 0)
        
        # 8. Усиление контраста результата
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=8)
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def _scale_image_for_ocr(self, image: np.ndarray, scale_factor: float = 3.0) -> np.ndarray:
        """Масштабирует изображение для OCR."""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

