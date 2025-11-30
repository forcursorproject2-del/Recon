"""
Базовый абстрактный класс для предобработки изображений.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class BasePreprocessor(ABC):
    """Абстрактный базовый класс для всех препроцессоров."""
    
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет предобработку изображения.
        
        Args:
            image: Входное изображение (BGR)
            
        Returns:
            Обработанное изображение (BGR)
        """
        pass
    
    @abstractmethod
    def get_variants(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Возвращает список вариантов предобработки.
        
        Args:
            image: Входное изображение
            
        Returns:
            Список кортежей (название_варианта, обработанное_изображение)
        """
        pass

