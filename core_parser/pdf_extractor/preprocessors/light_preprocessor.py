"""
Легкие варианты предобработки для хороших документов.

Использует существующий ImagePreprocessor для быстрой обработки.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

from .base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class LightPreprocessor(BasePreprocessor):
    """Класс для легкой предобработки хороших документов."""
    
    def __init__(self, image_preprocessor):
        """
        Инициализация.
        
        Args:
            image_preprocessor: Экземпляр ImagePreprocessor из image_preprocessor.py
        """
        self.image_preprocessor = image_preprocessor
    
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает 5 быстрых вариантов предобработки.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Список кортежей (название, обработанное_изображение)
        """
        variants = []
        
        # 1. Стандартная предобработка
        try:
            variants.append(('standard', self.image_preprocessor.preprocess(image, force=True)))
        except Exception as e:
            logger.warning(f"Ошибка standard: {e}")
        
        # 2. Агрессивная предобработка
        try:
            variants.append(('aggressive', self.image_preprocessor.preprocess_aggressive(image)))
        except Exception as e:
            logger.warning(f"Ошибка aggressive: {e}")
        
        # 3. Метод Оцу
        try:
            variants.append(('otsu', self.image_preprocessor.preprocess_otsu(image)))
        except Exception as e:
            logger.warning(f"Ошибка otsu: {e}")
        
        # 4. Морфологическая обработка
        try:
            variants.append(('morphology', self.image_preprocessor.preprocess_morphology(image)))
        except Exception as e:
            logger.warning(f"Ошибка morphology: {e}")
        
        # 5. Масштабированный оригинал
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            h, w = gray.shape
            if h < 1000 or w < 1000:
                scale = max(2000 / h, 2000 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            variants.append(('scaled_original', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
        except Exception as e:
            logger.warning(f"Ошибка scaled_original: {e}")
        
        return variants
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Базовый метод - использует стандартную предобработку."""
        return self.image_preprocessor.preprocess(image, force=True)

