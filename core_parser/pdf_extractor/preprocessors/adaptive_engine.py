"""
Адаптивный движок для определения уровня предобработки.

Анализирует качество изображения и начальный текст,
чтобы определить нужный уровень предобработки: light, medium или heavy.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

from .seal_remover import SealRemover

logger = logging.getLogger(__name__)


class AdaptivePreprocessingEngine:
    """Класс для адаптивного определения уровня предобработки."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.seal_remover = SealRemover(config=self.config)
    
    def get_preprocessing_level(self, image: np.ndarray, initial_text: str = "") -> Tuple[str, float]:
        """
        Определяет уровень предобработки на основе анализа изображения.
        
        Args:
            image: Входное изображение (BGR)
            initial_text: Текст, извлеченный без OCR (для оценки)
            
        Returns:
            Tuple[уровень, score]: ('light', 'medium' или 'heavy') и числовой score (0-100)
        """
        score = 0.0
        
        # 1. Яркость и контраст
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        brightness = np.mean(gray)
        contrast = gray.std()
        
        # Штраф за слишком темное или светлое
        if brightness < 100:
            score += (100 - brightness) * 0.6
        elif brightness > 200:
            score += (brightness - 200) * 0.4
        
        # Штраф за низкий контраст
        if contrast < 50:
            score += (50 - contrast) * 1.4
        
        # 2. Резкость (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            score += (100 - laplacian_var) * 0.9
        
        # 3. Мало текста без OCR → вероятно скан
        if len(initial_text.strip()) < 400:
            score += 45
        
        # 4. Цветные печати (синие/фиолетовые)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        if hsv is not None:
            blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
            purple_mask = cv2.inRange(hsv, (130, 50, 50), (160, 255, 255))
            seal_pixels = np.count_nonzero(blue_mask) + np.count_nonzero(purple_mask)
            if seal_pixels > 1500:
                score += 40
        
        # 5. Косой скан (по Hough lines)
        if self._is_skewed(image, threshold=2.0):
            score += 25
        
        # 6. Шум (по вариации локальных областей)
        noise_level = self._estimate_noise(gray)
        if noise_level > 0.15:
            score += noise_level * 100
        
        # Ограничиваем score
        score = min(100, score)
        
        # Определяем уровень
        if score < 35:
            level = "light"
        elif score < 70:
            level = "medium"
        else:
            level = "heavy"
        
        logger.debug(f"Адаптивный анализ: score={score:.1f}, level={level}")
        
        return level, score
    
    def _is_skewed(self, image: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Проверяет, является ли изображение косым.
        
        Args:
            image: Входное изображение
            threshold: Порог угла в градусах
            
        Returns:
            True если изображение косое
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Детекция краев
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Поиск линий
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return False
        
        # Вычисляем углы
        angles = []
        for line in lines[:20]:
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return False
        
        # Медианный угол
        median_angle = np.median(angles)
        
        return abs(median_angle) > threshold
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Оценивает уровень шума в изображении.
        
        Args:
            gray: Изображение в градациях серого
            
        Returns:
            Оценка уровня шума (0-1)
        """
        # Вычисляем вариацию в маленьких окнах
        h, w = gray.shape
        window_size = min(50, h // 10, w // 10)
        
        if window_size < 5:
            return 0.0
        
        variances = []
        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = gray[i:i+window_size, j:j+window_size]
                var = np.var(window)
                variances.append(var)
        
        if not variances:
            return 0.0
        
        # Высокая вариация = много шума
        avg_var = np.mean(variances)
        # Нормализуем (примерно)
        noise_level = min(1.0, avg_var / 1000.0)
        
        return noise_level

