"""
Мощные стратегии предобработки для плохих сканов.

Содержит 12 специализированных методов для обработки различных типов проблем:
- Серые документы
- Печати и подписи
- Косые сканы
- Шумные изображения
- Слабые линии таблиц
- Неравномерное освещение
- Размытые изображения
- Желтые/темные сканы
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

from .base_preprocessor import BasePreprocessor
from .seal_remover import SealRemover

logger = logging.getLogger(__name__)


class HeavyPreprocessor(BasePreprocessor):
    """Класс для мощной предобработки плохих сканов."""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.seal_remover = SealRemover(config=self.config)
    
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает все 12 вариантов тяжелой предобработки.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Список кортежей (название, обработанное_изображение)
        """
        variants = []
        
        # 1. CLAHE агрессивный
        try:
            variants.append(('clahe_aggressive', self._clahe_aggressive(image)))
        except Exception as e:
            logger.warning(f"Ошибка clahe_aggressive: {e}")
        
        # 2. Растяжение контраста
        try:
            variants.append(('contrast_stretch', self._contrast_stretch(image)))
        except Exception as e:
            logger.warning(f"Ошибка contrast_stretch: {e}")
        
        # 3. Удаление синих печатей
        try:
            variants.append(('remove_blue_seals', self._remove_blue_seals(image)))
        except Exception as e:
            logger.warning(f"Ошибка remove_blue_seals: {e}")
        
        # 4. Удаление фиолетовых печатей
        try:
            variants.append(('remove_purple_seals', self._remove_purple_seals(image)))
        except Exception as e:
            logger.warning(f"Ошибка remove_purple_seals: {e}")
        
        # 5. Точное выравнивание
        try:
            variants.append(('deskew_precise', self._deskew_precise(image)))
        except Exception as e:
            logger.warning(f"Ошибка deskew_precise: {e}")
        
        # 6. Сильное удаление шума
        try:
            variants.append(('denoise_strong', self._denoise_strong(image)))
        except Exception as e:
            logger.warning(f"Ошибка denoise_strong: {e}")
        
        # 7. Улучшение линий таблиц
        try:
            variants.append(('enhance_table_lines', self._enhance_table_lines(image)))
        except Exception as e:
            logger.warning(f"Ошибка enhance_table_lines: {e}")
        
        # 8. Адаптивная бинаризация (блок 41)
        try:
            variants.append(('adaptive_thresh_41', self._adaptive_thresh_41(image)))
        except Exception as e:
            logger.warning(f"Ошибка adaptive_thresh_41: {e}")
        
        # 9. Unsharp mask
        try:
            variants.append(('unsharp_mask', self._unsharp_mask(image)))
        except Exception as e:
            logger.warning(f"Ошибка unsharp_mask: {e}")
        
        # 10. Автоподбор гаммы
        try:
            variants.append(('gamma_auto', self._gamma_auto(image)))
        except Exception as e:
            logger.warning(f"Ошибка gamma_auto: {e}")
        
        # 11. Гибрид: печати + CLAHE + морфология
        try:
            variants.append(('hybrid_seal_clahe', self._hybrid_seal_clahe(image)))
        except Exception as e:
            logger.warning(f"Ошибка hybrid_seal_clahe: {e}")
        
        # 12. Ultimate combo (5 лучших в цепочке)
        try:
            variants.append(('ultimate_combo', self._ultimate_combo(image)))
        except Exception as e:
            logger.warning(f"Ошибка ultimate_combo: {e}")
        
        return variants
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Базовый метод - использует ultimate_combo."""
        return self._ultimate_combo(image)
    
    def _clahe_aggressive(self, image: np.ndarray) -> np.ndarray:
        """CLAHE с агрессивными параметрами для серых документов."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Агрессивный CLAHE
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _contrast_stretch(self, image: np.ndarray) -> np.ndarray:
        """Растяжение контраста для очень бледных документов."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Нормализация гистограммы
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Дополнительное растяжение
        alpha = 1.5  # Контраст
        beta = 0     # Яркость
        stretched = cv2.convertScaleAbs(normalized, alpha=alpha, beta=beta)
        
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    
    def _remove_blue_seals(self, image: np.ndarray) -> np.ndarray:
        """Удаление синих печатей."""
        return self.seal_remover.remove_blue_seals(image)
    
    def _remove_purple_seals(self, image: np.ndarray) -> np.ndarray:
        """Удаление фиолетовых печатей."""
        return self.seal_remover.remove_purple_seals(image)
    
    def _deskew_precise(self, image: np.ndarray) -> np.ndarray:
        """Точное выравнивание для косых сканов (до ±15°)."""
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
        for line in lines[:30]:  # Берем больше линий для точности
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            if -15 < angle < 15:  # Расширенный диапазон
                angles.append(angle)
        
        if not angles:
            return image
        
        # Медианный угол
        angle = np.median(angles)
        
        if abs(angle) < 0.3:  # Не поворачиваем если угол очень маленький
            return image
        
        # Поворот
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        
        logger.debug(f"Выравнивание: угол {angle:.2f}°")
        return rotated
    
    def _denoise_strong(self, image: np.ndarray) -> np.ndarray:
        """Сильное удаление шума для старых сканеров."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Медианный фильтр
        denoised = cv2.medianBlur(gray, 5)
        
        # Non-local means с агрессивными параметрами
        try:
            denoised = cv2.fastNlMeansDenoising(denoised, None, h=30,
                                                 templateWindowSize=7,
                                                 searchWindowSize=21)
        except Exception as e:
            logger.debug(f"FastNlMeansDenoising не доступен: {e}")
        
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    
    def _enhance_table_lines(self, image: np.ndarray) -> np.ndarray:
        """Улучшение слабых линий таблиц через морфологию."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Морфология для вертикальных линий
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)
        
        # Морфология для горизонтальных линий
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        
        # Объединяем
        table_lines = cv2.bitwise_or(vertical, horizontal)
        
        # Улучшаем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Комбинируем с линиями
        result = cv2.bitwise_and(enhanced, table_lines)
        result = cv2.addWeighted(enhanced, 0.7, result, 0.3, 0)
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    def _adaptive_thresh_41(self, image: np.ndarray) -> np.ndarray:
        """Адаптивная бинаризация с большим блоком для неравномерного освещения."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Адаптивная бинаризация с большим блоком
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41, 12  # Большой блок, больше C
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """Unsharp mask для размытых изображений."""
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    def _gamma_auto(self, image: np.ndarray) -> np.ndarray:
        """Автоподбор гаммы для желтых/темных сканов."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Вычисляем среднюю яркость
        mean_brightness = np.mean(gray)
        
        # Подбираем гамму
        if mean_brightness < 100:  # Темное
            gamma = 0.7
        elif mean_brightness > 180:  # Светлое/желтое
            gamma = 1.3
        else:
            gamma = 1.0
        
        # Применяем гамма-коррекцию
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        corrected = cv2.LUT(gray, table)
        
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    
    def _hybrid_seal_clahe(self, image: np.ndarray) -> np.ndarray:
        """Гибрид: удаление печатей + CLAHE + морфология."""
        # 1. Удаление печатей
        no_seals = self.seal_remover.remove_seals(image)
        
        # 2. CLAHE
        if len(no_seals.shape) == 3:
            gray = cv2.cvtColor(no_seals, cv2.COLOR_BGR2GRAY)
        else:
            gray = no_seals.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Морфология для улучшения текста
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
    
    def _ultimate_combo(self, image: np.ndarray) -> np.ndarray:
        """Ultimate combo: цепочка из 5 лучших методов."""
        # 1. Удаление печатей
        step1 = self.seal_remover.remove_seals(image)
        
        # 2. Выравнивание
        step2 = self._deskew_precise(step1)
        
        # 3. CLAHE агрессивный
        if len(step2.shape) == 3:
            gray = cv2.cvtColor(step2, cv2.COLOR_BGR2GRAY)
        else:
            gray = step2.copy()
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        step3 = clahe.apply(gray)
        
        # 4. Удаление шума
        try:
            step4 = cv2.fastNlMeansDenoising(step3, None, h=20,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
        except Exception:
            step4 = cv2.medianBlur(step3, 3)
        
        # 5. Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            step4, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 8
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

