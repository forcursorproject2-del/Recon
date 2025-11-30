"""
Модуль удаления цветных печатей и подписей из документов.

Использует HSV цветовое пространство для детекции синих и фиолетовых областей,
затем применяет inpainting для их удаления.
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SealRemover:
    """Класс для удаления печатей и подписей из документов."""
    
    def __init__(self, 
                 blue_range: tuple = None,
                 purple_range: tuple = None,
                 min_seal_area: int = None,
                 inpaint_radius: int = None,
                 inpaint_method: str = None,
                 config: dict = None):
        """
        Инициализация.
        
        Args:
            blue_range: Диапазон HSV для синих печатей ((h_min, s_min, v_min), (h_max, s_max, v_max))
            purple_range: Диапазон HSV для фиолетовых печатей
            min_seal_area: Минимальная площадь области для удаления (в пикселях)
            inpaint_radius: Радиус для inpainting
            inpaint_method: Метод inpainting ('telea' или 'ns')
            config: Словарь с настройками из config.yaml (приоритет над отдельными параметрами)
        """
        # Если передан config, используем его значения
        if config:
            seal_config = config.get('seal_removal', {})
            blue_cfg = seal_config.get('blue_range', {})
            purple_cfg = seal_config.get('purple_range', {})
            
            if blue_range is None:
                blue_range = (
                    (blue_cfg.get('h_min', 90), blue_cfg.get('s_min', 40), blue_cfg.get('v_min', 30)),
                    (blue_cfg.get('h_max', 130), blue_cfg.get('s_max', 255), blue_cfg.get('v_max', 255))
                )
            if purple_range is None:
                purple_range = (
                    (purple_cfg.get('h_min', 130), purple_cfg.get('s_min', 40), purple_cfg.get('v_min', 30)),
                    (purple_cfg.get('h_max', 160), purple_cfg.get('s_max', 255), purple_cfg.get('v_max', 255))
                )
            if min_seal_area is None:
                min_seal_area = seal_config.get('min_seal_area', 50)
            if inpaint_radius is None:
                inpaint_radius = seal_config.get('inpaint_radius', 5)
            if inpaint_method is None:
                inpaint_method = seal_config.get('inpaint_method', 'telea')
        
        # Значения по умолчанию, если не заданы
        if blue_range is None:
            blue_range = ((90, 50, 50), (130, 255, 255))
        if purple_range is None:
            purple_range = ((130, 50, 50), (160, 255, 255))
        if min_seal_area is None:
            min_seal_area = 100
        if inpaint_radius is None:
            inpaint_radius = 3
        if inpaint_method is None:
            inpaint_method = 'telea'
        
        self.blue_range = blue_range
        self.purple_range = purple_range
        self.min_seal_area = min_seal_area
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method
    
    def detect_seals(self, image: np.ndarray) -> np.ndarray:
        """
        Обнаруживает области печатей и подписей.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Бинарная маска областей печатей
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Маска для синих печатей
        lower_blue = np.array(self.blue_range[0])
        upper_blue = np.array(self.blue_range[1])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Маска для фиолетовых печатей
        lower_purple = np.array(self.purple_range[0])
        upper_purple = np.array(self.purple_range[1])
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Объединяем маски
        mask = cv2.bitwise_or(mask_blue, mask_purple)
        
        # Морфологическая обработка для удаления шума
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Удаляем маленькие области
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_cleaned = np.zeros_like(mask)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_seal_area:
                cv2.drawContours(mask_cleaned, [contour], -1, 255, -1)
        
        return mask_cleaned
    
    def remove_seals(self, image: np.ndarray, method: str = 'telea') -> np.ndarray:
        """
        Удаляет печати и подписи из изображения.
        
        Args:
            image: Входное изображение (BGR)
            method: Метод inpainting ('telea' или 'ns')
            
        Returns:
            Изображение с удаленными печатями
        """
        mask = self.detect_seals(image)
        
        if np.count_nonzero(mask) == 0:
            logger.debug("Печати не обнаружены")
            return image
        
        # Используем метод из параметра или из конфига
        actual_method = method if method else self.inpaint_method
        inpaint_method = cv2.INPAINT_TELEA if actual_method == 'telea' else cv2.INPAINT_NS
        
        # Применяем inpainting с радиусом из конфига
        result = cv2.inpaint(image, mask, self.inpaint_radius, inpaint_method)
        
        seal_pixels = np.count_nonzero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        seal_percentage = (seal_pixels / total_pixels) * 100
        
        logger.debug(f"Удалено печатей: {seal_pixels} пикселей ({seal_percentage:.2f}%)")
        
        return result
    
    def remove_blue_seals(self, image: np.ndarray) -> np.ndarray:
        """Удаляет только синие печати."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Расширенный диапазон для синих печатей (более агрессивно)
        lower_blue1 = np.array([90, 30, 20])  # Более широкий диапазон
        upper_blue1 = np.array([130, 255, 255])
        mask1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
        
        # Дополнительный диапазон для темно-синих
        lower_blue2 = np.array([100, 40, 30])
        upper_blue2 = np.array([120, 255, 255])
        mask2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
        
        # Объединяем маски
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Морфология (более агрессивно)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        # Удаляем маленькие области
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_seal_area:
                cv2.drawContours(mask_cleaned, [contour], -1, 255, -1)
        
        if np.count_nonzero(mask_cleaned) == 0:
            return image
        
        inpaint_method = cv2.INPAINT_TELEA if self.inpaint_method == 'telea' else cv2.INPAINT_NS
        # Увеличиваем радиус для лучшего закрашивания
        return cv2.inpaint(image, mask_cleaned, self.inpaint_radius + 2, inpaint_method)
    
    def remove_purple_seals(self, image: np.ndarray) -> np.ndarray:
        """Удаляет только фиолетовые печати."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_purple = np.array(self.purple_range[0])
        upper_purple = np.array(self.purple_range[1])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        # Морфология
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Удаляем маленькие области
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_cleaned = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_seal_area:
                cv2.drawContours(mask_cleaned, [contour], -1, 255, -1)
        
        inpaint_method = cv2.INPAINT_TELEA if self.inpaint_method == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask_cleaned, self.inpaint_radius, inpaint_method)

