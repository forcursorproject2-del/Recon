"""
Специализированный препроцессор для фото и PDF с фотографиями.

Использует единый стабильный пайплайн предобработки на основе лучших практик
для OCR с русским языком:
- Обязательная конвертация в чёрно-белый формат
- CLAHE для улучшения контраста
- Адаптивная бинаризация
- Удаление шума
- Выравнивание документа
- Масштабирование при необходимости

Не использует множественные варианты - один надёжный пайплайн.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List

from .base_preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)


class PhotoPreprocessor(BasePreprocessor):
    """
    Специализированный препроцессор для фото и PDF с фотографиями.
    
    Использует единый стабильный пайплайн без вариантов.
    """
    
    def __init__(self, config: dict = None):
        """
        Инициализация препроцессора для фото.
        
        Args:
            config: Словарь с настройками
        """
        self.config = config or {}
        
        # Параметры CLAHE (более агрессивные для фото)
        self.clahe_clip_limit = self.config.get('clahe_clip_limit', 4.0)
        self.clahe_tile_size = self.config.get('clahe_tile_size', 8)
        
        # Минимальный размер для масштабирования (увеличиваем для лучшего распознавания)
        self.min_size_for_scale = self.config.get('min_size_for_scale', 1500)
        self.target_size = self.config.get('target_size', 3000)
        
        # Порог для определения необходимости выравнивания
        self.deskew_threshold = self.config.get('deskew_threshold', 0.5)
        
        # Параметры удаления шума (мягкие, чтобы не потерять детали)
        self.denoise_h = self.config.get('denoise_h', 8)
        self.denoise_template_size = self.config.get('denoise_template_size', 7)
        self.denoise_search_size = self.config.get('denoise_search_size', 21)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Единый стабильный пайплайн предобработки для фото.
        
        Оптимизирован для PaddleOCR - работает лучше с grayscale, чем с бинаризацией.
        
        Последовательность:
        1. Конвертация в чёрно-белый (grayscale)
        2. Масштабирование при необходимости (увеличение для лучшего распознавания)
        3. Выравнивание (deskew)
        4. Улучшение контраста (CLAHE) - агрессивное для фото
        5. Удаление шума (мягкое, чтобы не потерять детали)
        6. Улучшение резкости (для лучшего распознавания текста)
        7. Нормализация яркости
        8. Конвертация обратно в BGR для совместимости
        
        Args:
            image: Входное изображение (BGR или RGB)
            
        Returns:
            Обработанное изображение (BGR) - grayscale с улучшенным контрастом
        """
        try:
            # Шаг 1: Конвертация в grayscale (обязательно)
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                elif image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]  # Берём первый канал
            else:
                gray = image.copy()
            
            logger.debug(f"Конвертировано в grayscale: {gray.shape}")
            
            # Шаг 2: Масштабирование при необходимости (увеличиваем для лучшего распознавания)
            gray = self._scale_if_needed(gray)
            
            # Шаг 3: Выравнивание документа
            gray = self._deskew_image(gray)
            
            # Шаг 4: Улучшение контраста через CLAHE (агрессивное для фото)
            gray = self._enhance_contrast_clahe(gray)
            
            # Шаг 5: Удаление шума (мягкое, чтобы сохранить детали текста)
            gray = self._remove_noise(gray)
            
            # Шаг 6: Улучшение резкости (важно для PaddleOCR)
            gray = self._sharpen_image(gray)
            
            # Шаг 7: Нормализация яркости (убираем слишком тёмные/светлые области)
            gray = self._normalize_brightness(gray)
            
            # Шаг 8: Конвертация обратно в BGR для совместимости с OCR
            # PaddleOCR лучше работает с grayscale, но принимает BGR
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            logger.debug("Предобработка фото завершена успешно")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при предобработке фото: {e}", exc_info=True)
            # В случае ошибки возвращаем хотя бы grayscale версию
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает только один вариант - единый стабильный пайплайн.
        
        Для фото не используем множественные варианты.
        """
        return [('photo_pipeline', self.preprocess(image))]
    
    def _scale_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        Масштабирует изображение если оно слишком маленькое.
        
        Для OCR лучше работать с изображениями достаточного размера.
        """
        h, w = image.shape[:2]
        
        if h < self.min_size_for_scale or w < self.min_size_for_scale:
            # Вычисляем коэффициент масштабирования
            scale_h = self.target_size / h if h < self.min_size_for_scale else 1.0
            scale_w = self.target_size / w if w < self.min_size_for_scale else 1.0
            scale = max(scale_h, scale_w)
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Используем INTER_CUBIC для лучшего качества при увеличении
            scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            logger.debug(f"Масштабирование: {w}x{h} -> {new_w}x{new_h} (scale={scale:.2f})")
            return scaled
        
        return image
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Выравнивает изображение (убирает наклон).
        
        Использует метод Hough Lines для определения угла наклона.
        """
        # Детекция краев
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Поиск линий
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Вычисляем углы наклона
        angles = []
        for line in lines[:30]:  # Берём первые 30 линий
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            # Учитываем только горизонтальные линии (игнорируем вертикальные)
            if -45 < angle < 45:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Используем медианный угол для устойчивости к выбросам
        median_angle = np.median(angles)
        
        # Если угол слишком маленький, не поворачиваем
        if abs(median_angle) < self.deskew_threshold:
            return image
        
        logger.debug(f"Выравнивание изображения: угол {median_angle:.2f}°")
        
        # Поворачиваем изображение
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _enhance_contrast_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшает контраст через CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE особенно эффективен для документов с неравномерным освещением,
        что часто встречается в фото.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Удаляет шум из изображения.
        
        Использует комбинацию медианного фильтра и non-local means denoising.
        """
        # Сначала медианный фильтр для удаления солевого шума
        denoised = cv2.medianBlur(image, 3)
        
        # Затем non-local means denoising для сохранения деталей текста
        try:
            denoised = cv2.fastNlMeansDenoising(
                denoised,
                None,
                h=self.denoise_h,
                templateWindowSize=self.denoise_template_size,
                searchWindowSize=self.denoise_search_size
            )
        except Exception as e:
            logger.debug(f"FastNlMeansDenoising не доступен: {e}, используем только медианный фильтр")
        
        return denoised
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшает резкость изображения.
        
        Важно для PaddleOCR - лучше распознаёт чёткий текст.
        """
        # Unsharp masking для улучшения резкости
        # Сначала размываем
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Затем вычитаем размытое из оригинала и добавляем обратно
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        # Ограничиваем значения в диапазоне [0, 255]
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Нормализует яркость изображения.
        
        Убирает слишком тёмные и слишком светлые области,
        улучшая общий контраст для OCR.
        """
        # Вычисляем среднюю яркость
        mean_brightness = np.mean(image)
        
        # Если изображение слишком тёмное или светлое, корректируем
        if mean_brightness < 80:
            # Слишком тёмное - осветляем
            alpha = 1.2
            beta = 20
            normalized = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif mean_brightness > 180:
            # Слишком светлое - затемняем
            alpha = 0.9
            beta = -10
            normalized = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        else:
            # Нормальная яркость - только небольшая коррекция контраста
            normalized = image.copy()
        
        # Дополнительная нормализация гистограммы для улучшения контраста
        normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def is_photo(self, image: np.ndarray, initial_text: str = "") -> Tuple[bool, float]:
        """
        Определяет, является ли изображение фото (а не сканом).
        
        Критерии:
        - Мало текста без OCR (< 300 символов)
        - Низкая резкость
        - Низкий контраст
        - Наличие цветных областей (не только текст)
        
        Args:
            image: Входное изображение
            initial_text: Текст, извлеченный без OCR
            
        Returns:
            Tuple[is_photo, confidence]: (bool, float 0-1)
        """
        score = 0.0
        
        # Критерий 1: Мало текста без OCR
        if len(initial_text.strip()) < 300:
            score += 0.3
        
        # Критерий 2: Низкая резкость (Laplacian variance)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            score += 0.25
        
        # Критерий 3: Низкий контраст
        contrast = gray.std()
        if contrast < 50:
            score += 0.25
        
        # Критерий 4: Наличие цветных областей (не только текст)
        if len(image.shape) == 3:
            # Вычисляем стандартное отклонение по каналам
            # Если есть цвет, то std будет выше
            color_variance = np.std(image, axis=2).mean()
            if color_variance > 20:  # Есть цветные области
                score += 0.2
        
        confidence = min(1.0, score)
        is_photo = confidence > 0.5
        
        logger.debug(f"Определение фото: is_photo={is_photo}, confidence={confidence:.2f}")
        
        return is_photo, confidence

