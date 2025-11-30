"""
Модуль предобработки изображений для OCR.

Содержит три уровня предобработки:
- light: быстрые варианты для хороших документов
- medium: средние варианты для документов среднего качества
- heavy: мощные стратегии для плохих сканов
"""

from .base_preprocessor import BasePreprocessor
from .light_preprocessor import LightPreprocessor
from .heavy_preprocessor import HeavyPreprocessor
from .adaptive_engine import AdaptivePreprocessingEngine
from .seal_remover import SealRemover
from .photo_preprocessor import PhotoPreprocessor

__all__ = [
    'BasePreprocessor',
    'LightPreprocessor',
    'HeavyPreprocessor',
    'AdaptivePreprocessingEngine',
    'SealRemover',
    'PhotoPreprocessor',
]

