"""
Core Parser - Система автоматического парсинга финансовых документов.

Основные модули:
- config_manager: Управление конфигурацией
- pdf_extractor: Извлечение текста из PDF с поддержкой OCR
- classifier: Классификация документов с использованием ансамбля методов
- semantic_parser: Семантический анализ и извлечение полей
- table_builder: Нормализация и обработка таблиц
- batch_processor: Многопоточная обработка документов
- learning_engine: Активное обучение и дообучение модели
"""

__version__ = "2.1.0"
__author__ = "Core Parser Team"

from core_parser.config_manager.config_loader import ConfigManager
from core_parser.batch_processor.pipeline import BatchProcessingPipeline
from core_parser.classifier.document_classifier import DocumentClassifier, ClassificationResult
from core_parser.semantic_parser.field_extractors import SemanticParser, FieldExtractor, ExtractedField
from core_parser.table_builder.table_normalizer import TableBuilder
from core_parser.learning_engine.trainer import LearningEngine

__all__ = [
    'ConfigManager',
    'BatchProcessingPipeline',
    'DocumentClassifier',
    'ClassificationResult',
    'SemanticParser',
    'FieldExtractor',
    'ExtractedField',
    'TableBuilder',
    'LearningEngine',
]

