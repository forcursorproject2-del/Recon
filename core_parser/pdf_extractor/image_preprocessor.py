"""
Модуль предобработки изображений для улучшения качества OCR.

Использует OpenCV для локальной обработки без интернета:
- Выравнивание и де-скью документов
- Адаптивная бинаризация
- Увеличение контрастности
- Удаление шума
- Улучшение резкости
- Автоматическое определение необходимости предобработки
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Класс для предобработки изображений перед OCR."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация препроцессора.
        
        Args:
            config: Словарь с настройками предобработки
        """
        self.config = config or {}
        
        # Настройки по умолчанию
        self.enable_adaptive_threshold = self.config.get('adaptive_threshold', True)
        self.enable_deskew = self.config.get('deskew', True)
        self.enable_contrast_enhancement = self.config.get('contrast_enhancement', True)
        self.enable_noise_removal = self.config.get('noise_removal', True)
        self.enable_sharpening = self.config.get('sharpening', True)
        self.auto_detect_quality = self.config.get('auto_detect_quality', True)
        
        # Пороги для автоматического определения необходимости обработки
        self.min_blur_threshold = self.config.get('min_blur_threshold', 100.0)
        self.min_contrast_threshold = self.config.get('min_contrast_threshold', 20.0)
    
    def needs_preprocessing(self, image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """
        Определяет, нужна ли предобработка изображения.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Tuple[bool, Dict]: (нужна ли обработка, метрики качества)
        """
        if not self.auto_detect_quality:
            return True, {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Метрики качества
        metrics = {
            'blur': self._measure_blur(gray),
            'contrast': self._measure_contrast(gray),
            'brightness': np.mean(gray),
        }
        
        # Определяем необходимость обработки
        needs = (
            metrics['blur'] < self.min_blur_threshold or
            metrics['contrast'] < self.min_contrast_threshold or
            metrics['brightness'] < 50 or metrics['brightness'] > 200
        )
        
        logger.debug(f"Метрики качества изображения: {metrics}, нужна обработка: {needs}")
        return needs, metrics
    
    def _measure_blur(self, image: np.ndarray) -> float:
        """Измеряет размытость изображения (метод Лапласа)."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return float(np.var(laplacian))
    
    def _measure_contrast(self, image: np.ndarray) -> float:
        """Измеряет контрастность изображения."""
        return float(np.std(image))
    
    def preprocess(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Выполняет предобработку изображения.
        
        Args:
            image: Входное изображение (BGR или Grayscale)
            force: Принудительная обработка даже если метрики хорошие
            
        Returns:
            Обработанное изображение (BGR)
        """
        # Проверяем, нужно ли обрабатывать
        if not force:
            needs, metrics = self.needs_preprocessing(image)
            if not needs:
                logger.debug("Изображение не требует предобработки")
                return image
        
        # Конвертируем в RGB для работы
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            processed = image.copy()
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        original = processed.copy()
        
        # 1. Конвертация в grayscale для большинства операций
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 2. Выравнивание (де-скью)
        if self.enable_deskew:
            gray = self._deskew_image(gray)
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 3. Улучшение контрастности
        if self.enable_contrast_enhancement:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = self._enhance_contrast(processed, gray)
        
        # 4. Удаление шума
        if self.enable_noise_removal:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            denoised = self._remove_noise(gray)
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        # 5. Адаптивная бинаризация (для лучшего распознавания текста)
        if self.enable_adaptive_threshold:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            binary = self._adaptive_threshold(gray)
            processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # 6. Улучшение резкости
        if self.enable_sharpening:
            processed = self._sharpen_image(processed)
        
        logger.debug("Предобработка изображения завершена")
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Выравнивает изображение (убирает наклон/скью).
        
        Использует метод проекции для определения угла наклона.
        """
        # Детекция краев для лучшего определения угла
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Находим линии
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None or len(lines) == 0:
            return image
        
        # Вычисляем средний угол наклона
        angles = []
        # cv2.HoughLines возвращает массив вида [[rho, theta], [rho, theta], ...]
        for line in lines[:20]:  # Берем первые 20 линий
            if line is None or len(line) < 2:
                continue
            rho, theta = line[0], line[1]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:  # Игнорируем вертикальные линии
                angles.append(angle)
        
        if not angles:
            return image
        
        # Медианный угол для устойчивости к выбросам
        angle = np.median(angles)
        
        # Если угол небольшой (< 1 градус), не поворачиваем
        if abs(angle) < 0.5:
            return image
        
        logger.debug(f"Выравнивание изображения: угол {angle:.2f}°")
        
        # Поворачиваем изображение
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _enhance_contrast(self, image: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Улучшает контрастность изображения.
        
        Использует CLAHE (Contrast Limited Adaptive Histogram Equalization).
        """
        # Применяем CLAHE к каждому каналу
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(image.shape) == 3:
            # Обрабатываем каждый канал отдельно
            enhanced_channels = []
            for i in range(image.shape[2]):
                enhanced_channels.append(clahe.apply(image[:, :, i]))
            enhanced = np.stack(enhanced_channels, axis=2)
        else:
            enhanced = clahe.apply(gray)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Удаляет шум из изображения.
        
        Использует комбинацию методов для сохранения текста.
        """
        # Медианный фильтр для удаления солевого шума
        denoised = cv2.medianBlur(image, 3)
        
        # Non-local means denoising для сохранения деталей
        # (используем быструю версию)
        try:
            denoised = cv2.fastNlMeansDenoising(denoised, None, h=10, 
                                                 templateWindowSize=7, 
                                                 searchWindowSize=21)
        except Exception as e:
            logger.debug(f"FastNlMeansDenoising не доступен: {e}")
        
        return denoised
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Применяет адаптивную бинаризацию.
        
        Лучше работает с документами с неравномерным освещением.
        """
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return binary
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Улучшает резкость изображения.
        
        Использует unsharp masking для улучшения читаемости текста.
        """
        # Ядро для повышения резкости (unsharp mask)
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Смешиваем оригинал и обработанное изображение (70% sharpened, 30% original)
        result = cv2.addWeighted(sharpened, 0.7, image, 0.3, 0)
        
        return result
    
    def preprocess_fast(self, image: np.ndarray) -> np.ndarray:
        """
        Быстрая предобработка для хороших изображений.
        
        Применяет только минимальные улучшения.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Только адаптивная бинаризация и небольшое улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def preprocess_aggressive(self, image: np.ndarray) -> np.ndarray:
        """
        Агрессивная предобработка для очень плохих изображений.
        
        Применяет усиленные методы улучшения качества.
        """
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Масштабирование для лучшего распознавания (увеличиваем в 2 раза если маленькое)
        h, w = gray.shape
        if h < 1000 or w < 1000:
            scale = max(2000 / h, 2000 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Масштабирование изображения: {w}x{h} -> {new_w}x{new_h}")
        
        # 2. Улучшение яркости и контраста (более агрессивное)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Удаление шума (более агрессивное)
        denoised = cv2.medianBlur(enhanced, 5)
        try:
            denoised = cv2.fastNlMeansDenoising(denoised, None, h=15, 
                                                 templateWindowSize=7, 
                                                 searchWindowSize=21)
        except Exception:
            pass
        
        # 4. Морфологическая обработка для улучшения текста
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # 5. Адаптивная бинаризация с разными параметрами
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 5
        )
        
        # 6. Улучшение резкости
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        sharpened = cv2.filter2D(binary, -1, kernel_sharpen)
        
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    def preprocess_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка с использованием метода Оцу для бинаризации.
        
        Хорошо работает для изображений с хорошим контрастом.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Масштабирование если нужно
        h, w = gray.shape
        if h < 1000 or w < 1000:
            scale = max(2000 / h, 2000 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Бинаризация Оцу
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def preprocess_morphology(self, image: np.ndarray) -> np.ndarray:
        """
        Предобработка с использованием морфологических операций.
        
        Хорошо работает для текста с артефактами.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Масштабирование
        h, w = gray.shape
        if h < 1000 or w < 1000:
            scale = max(2000 / h, 2000 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Морфологическое закрытие для соединения разорванных символов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            closed, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def preprocess_multiple_variants(self, image: np.ndarray) -> list:
        """
        Генерирует несколько вариантов предобработанных изображений.
        
        Returns:
            Список обработанных изображений с описаниями
        """
        variants = []
        
        # Вариант 1: Стандартная предобработка
        try:
            variant1 = self.preprocess(image, force=True)
            variants.append(('standard', variant1))
        except Exception as e:
            logger.warning(f"Ошибка стандартной предобработки: {e}")
        
        # Вариант 2: Агрессивная предобработка
        try:
            variant2 = self.preprocess_aggressive(image)
            variants.append(('aggressive', variant2))
        except Exception as e:
            logger.warning(f"Ошибка агрессивной предобработки: {e}")
        
        # Вариант 3: Метод Оцу
        try:
            variant3 = self.preprocess_otsu(image)
            variants.append(('otsu', variant3))
        except Exception as e:
            logger.warning(f"Ошибка предобработки Оцу: {e}")
        
        # Вариант 4: Морфологическая обработка
        try:
            variant4 = self.preprocess_morphology(image)
            variants.append(('morphology', variant4))
        except Exception as e:
            logger.warning(f"Ошибка морфологической предобработки: {e}")
        
        # Вариант 5: Без предобработки (оригинал, но масштабированный)
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
            logger.warning(f"Ошибка масштабирования оригинала: {e}")
        
        return variants

