"""
Универсальный препроцессор с детекцией типа деградации.

Определяет тип деградации изображения и применяет специализированную обработку.
Работает универсально для всех типов документов.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, List

from .base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class UniversalPreprocessor(BasePreprocessor):
    """
    Универсальный препроцессор, который определяет тип деградации
    и применяет специализированную обработку.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Универсальная предобработка с автоматической детекцией типа деградации.
        """
        degradation_metrics = self.detect_degradation_type(image)
        return self.preprocess_for_degradation(image, degradation_metrics)
    
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает варианты предобработки для разных типов деградации.
        """
        variants = []
        degradation_metrics = self.detect_degradation_type(image)
        
        # Базовый вариант - универсальная обработка
        try:
            variants.append(('universal', self.preprocess_for_degradation(image, degradation_metrics)))
        except Exception as e:
            logger.warning(f"Ошибка universal preprocessing: {e}")
        
        # Специализированные варианты в зависимости от типа деградации
        if degradation_metrics['blur'] < 50:
            try:
                variants.append(('deblur', self._deblur_image(image)))
            except Exception as e:
                logger.warning(f"Ошибка deblur: {e}")
        
        if degradation_metrics['noise'] > 0.2:
            try:
                variants.append(('aggressive_denoise', self._aggressive_denoise(image)))
            except Exception as e:
                logger.warning(f"Ошибка aggressive_denoise: {e}")
        
        if degradation_metrics['low_contrast'] < 30:
            try:
                variants.append(('enhance_contrast_aggressive', self._enhance_contrast_aggressive(image)))
            except Exception as e:
                logger.warning(f"Ошибка enhance_contrast_aggressive: {e}")
        
        if degradation_metrics['skew'] > 2.0:
            try:
                variants.append(('deskew_precise', self._deskew_precise(image)))
            except Exception as e:
                logger.warning(f"Ошибка deskew_precise: {e}")
        
        if degradation_metrics['uneven_lighting'] > 0.3:
            try:
                variants.append(('correct_lighting', self._correct_uneven_lighting(image)))
            except Exception as e:
                logger.warning(f"Ошибка correct_lighting: {e}")
        
        # Всегда добавляем базовый вариант
        if not variants:
            variants.append(('standard', image))
        
        return variants
    
    def detect_degradation_type(self, image: np.ndarray) -> Dict[str, float]:
        """
        Определяет тип деградации изображения.
        
        Returns:
            Словарь с оценками различных типов деградации (0-1, где 1 = максимальная деградация)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Нормализуем метрики к диапазону 0-1
        blur = self._measure_blur(gray)
        blur_normalized = min(1.0, max(0.0, 1.0 - (blur / 500.0)))  # Нормализуем
        
        noise = self._measure_noise(gray)
        noise_normalized = min(1.0, noise)
        
        contrast = self._measure_contrast(gray)
        low_contrast_normalized = min(1.0, max(0.0, 1.0 - (contrast / 100.0)))
        
        skew = self._measure_skew(gray)
        skew_normalized = min(1.0, abs(skew) / 10.0)  # Нормализуем угол
        
        uneven_lighting = self._measure_lighting_uniformity(gray)
        lighting_normalized = min(1.0, uneven_lighting)
        
        color_cast = 0.0
        if len(image.shape) == 3:
            color_cast = self._measure_color_cast(image)
        
        metrics = {
            'blur': blur_normalized,
            'noise': noise_normalized,
            'low_contrast': low_contrast_normalized,
            'skew': skew_normalized,
            'uneven_lighting': lighting_normalized,
            'color_cast': color_cast
        }
        
        logger.debug(f"Метрики деградации: {metrics}")
        return metrics
    
    def preprocess_for_degradation(self, image: np.ndarray, degradation_metrics: Dict[str, float]) -> np.ndarray:
        """
        Применяет специализированную предобработку в зависимости от типа деградации.
        """
        processed = image.copy()
        
        # Приоритет обработки по типу деградации
        if degradation_metrics['blur'] > 0.5:
            # Сильное размытие - деконволюция и резкость
            processed = self._deblur_image(processed)
        
        if degradation_metrics['noise'] > 0.2:
            # Много шума - агрессивное удаление шума
            processed = self._aggressive_denoise(processed)
        
        if degradation_metrics['low_contrast'] > 0.5:
            # Низкий контраст - усиленное улучшение контраста
            processed = self._enhance_contrast_aggressive(processed)
        
        if degradation_metrics['skew'] > 0.2:
            # Косой скан - выравнивание
            processed = self._deskew_precise(processed)
        
        if degradation_metrics['uneven_lighting'] > 0.3:
            # Неравномерное освещение - адаптивная коррекция
            processed = self._correct_uneven_lighting(processed)
        
        # Всегда применяем базовую обработку
        processed = self._apply_base_preprocessing(processed)
        
        return processed
    
    def _apply_base_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Базовая предобработка для всех изображений."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _measure_blur(self, gray: np.ndarray) -> float:
        """Измеряет размытость (Laplacian variance)."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.var(laplacian))
    
    def _measure_noise(self, gray: np.ndarray) -> float:
        """Оценивает уровень шума."""
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
        
        avg_var = np.mean(variances)
        # Нормализуем (примерно)
        noise_level = min(1.0, avg_var / 1000.0)
        
        return noise_level
    
    def _measure_contrast(self, gray: np.ndarray) -> float:
        """Измеряет контрастность."""
        return float(np.std(gray))
    
    def _measure_skew(self, gray: np.ndarray) -> float:
        """Измеряет угол наклона в градусах."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        angles = []
        for line in lines[:20]:
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        return float(np.median(angles))
    
    def _measure_lighting_uniformity(self, gray: np.ndarray) -> float:
        """Измеряет равномерность освещения."""
        h, w = gray.shape
        # Делим на 4 области и сравниваем яркость
        h_mid = h // 2
        w_mid = w // 2
        
        regions = [
            gray[0:h_mid, 0:w_mid],
            gray[0:h_mid, w_mid:w],
            gray[h_mid:h, 0:w_mid],
            gray[h_mid:h, w_mid:w]
        ]
        
        means = [np.mean(region) for region in regions]
        std_of_means = np.std(means)
        
        # Нормализуем
        uniformity = min(1.0, std_of_means / 50.0)
        
        return uniformity
    
    def _measure_color_cast(self, image: np.ndarray) -> float:
        """Измеряет цветовой сдвиг (для цветных изображений)."""
        if len(image.shape) != 3:
            return 0.0
        
        # Проверяем отклонение от нейтрального серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)
        
        # Вычисляем средние значения каналов
        mean_b = np.mean(b)
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        mean_gray = np.mean(gray)
        
        # Отклонение от серого
        deviation = np.sqrt(
            (mean_b - mean_gray)**2 + 
            (mean_g - mean_gray)**2 + 
            (mean_r - mean_gray)**2
        ) / 255.0
        
        return min(1.0, deviation)
    
    def _deblur_image(self, image: np.ndarray) -> np.ndarray:
        """Улучшает размытое изображение."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Unsharp masking для улучшения резкости
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Дополнительное улучшение резкости
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def _aggressive_denoise(self, image: np.ndarray) -> np.ndarray:
        """Агрессивное удаление шума."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Медианный фильтр
        denoised = cv2.medianBlur(gray, 5)
        
        # Non-local means denoising
        try:
            denoised = cv2.fastNlMeansDenoising(denoised, None, h=15,
                                                 templateWindowSize=7,
                                                 searchWindowSize=21)
        except Exception:
            pass
        
        # Морфологическое закрытие для соединения разорванных символов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _enhance_contrast_aggressive(self, image: np.ndarray) -> np.ndarray:
        """Агрессивное улучшение контраста."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # CLAHE с высоким clipLimit
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Растяжение гистограммы
        min_val = np.percentile(enhanced, 2)
        max_val = np.percentile(enhanced, 98)
        stretched = np.clip((enhanced - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    
    def _deskew_precise(self, image: np.ndarray) -> np.ndarray:
        """Точное выравнивание изображения."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Детекция краев
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Поиск линий
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Вычисляем углы
        angles = []
        for line in lines[:30]:
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Медианный угол
        angle = np.median(angles)
        
        if abs(angle) < 0.5:
            return image
        
        # Поворачиваем
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _correct_uneven_lighting(self, image: np.ndarray) -> np.ndarray:
        """Корректирует неравномерное освещение."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Создаем модель освещения (размытое изображение)
        lighting_model = cv2.GaussianBlur(gray, (0, 0), 50)
        
        # Нормализуем относительно модели освещения
        normalized = cv2.divide(gray, lighting_model, scale=255)
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

