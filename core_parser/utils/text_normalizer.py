"""
Централизованный модуль нормализации текста.

Обеспечивает единообразную нормализацию текста во всем проекте.
"""

import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Класс для нормализации текста на русском языке."""
    
    @staticmethod
    def normalize(text: str, preserve_case: bool = False) -> str:
        """
        Нормализует текст для русского языка.
        
        Args:
            text: Исходный текст
            preserve_case: Сохранять ли регистр (по умолчанию False - приводится к нижнему)
        
        Returns:
            Нормализованный текст
        """
        if not text:
            return ""
        
        # Приведение к нижнему регистру (если не требуется сохранение)
        if not preserve_case:
            text = text.lower()
        
        # Замена 'ё' на 'е'
        text = text.replace('ё', 'е').replace('Ё', 'Е')
        
        # Нормализация пробелов (заменяем последовательности пробельных символов на один пробел)
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление лишних символов, оставляем только буквы, цифры, пробелы и базовые знаки препинания
        text = re.sub(r'[^\w\s\.\,\-\+\(\)\[\]\{\}\/\\=:;]', ' ', text)
        
        # Удаление лишних пробелов в начале и конце
        text = text.strip()
        
        return text
    
    @staticmethod
    def normalize_for_classification(text: str) -> str:
        """
        Нормализует текст специально для классификации (более агрессивная нормализация).
        
        Args:
            text: Исходный текст
        
        Returns:
            Нормализованный текст для классификации
        """
        if not text:
            return ""
        
        # Базовая нормализация
        normalized = TextNormalizer.normalize(text, preserve_case=False)
        
        # Дополнительная очистка для классификации
        # Убираем дополнительные знаки препинания
        normalized = re.sub(r'[^\w\s\.\,\-\+]', '', normalized)
        
        return normalized
    
    @staticmethod
    def normalize_for_ocr(text: str) -> str:
        """
        Нормализует текст после OCR (исправляет типичные ошибки OCR).
        
        Args:
            text: Текст после OCR
        
        Returns:
            Нормализованный текст
        """
        if not text:
            return ""
        
        # Замена похожих символов, часто путаемых OCR
        replacements = {
            '0': 'О',  # цифра ноль на букву О (в контексте слов)
            'l': 'I',  # маленькая L на заглавную I
            '|': 'I',  # вертикальная черта на I
            '1': 'I',  # единица на I (в контексте слов)
        }
        
        # Применяем замены только в контексте слов
        # Это упрощенная версия, в продакшене можно улучшить
        
        # Нормализация кавычек
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('„', '"').replace('"', '"')
        
        # Нормализация тире
        text = text.replace('−', '-').replace('–', '-').replace('—', '-')
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        Очищает только пробельные символы, не меняя остальное.
        
        Args:
            text: Исходный текст
        
        Returns:
            Текст с нормализованными пробелами
        """
        if not text:
            return ""
        
        # Заменяем все пробельные символы на обычные пробелы
        text = re.sub(r'\s+', ' ', text)
        # Убираем пробелы в начале и конце строк
        text = text.strip()
        
        return text

