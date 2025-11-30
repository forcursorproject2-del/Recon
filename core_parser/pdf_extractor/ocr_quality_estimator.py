"""
Модуль для оценки качества распознанного текста и комбинирования результатов.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
import difflib

logger = logging.getLogger(__name__)


class OCRQualityEstimator:
    """Класс для оценки качества распознанного текста."""
    
    def __init__(self):
        # Паттерны для определения мусора
        self.garbage_patterns = [
            r'^[A-Za-zА-Яа-я]{1,2}\s*$',  # Одиночные буквы
            r'^[0-9]{1,2}\s*$',  # Одиночные цифры
            r'^[^\w\s]{1,3}\s*$',  # Только знаки препинания
        ]
        
        # Минимальные длины для разных типов контента
        self.min_word_length = 2
        self.min_sentence_length = 10
    
    def estimate_text_quality(self, text: str, reference: str = "") -> float:
        """
        Оценивает качество распознанного текста.
        
        Args:
            text: Распознанный текст
            reference: Референсный текст для сравнения (если есть)
            
        Returns:
            Оценка качества от 0 до 1
        """
        if not text or not text.strip():
            return 0.0
        
        text = text.strip()
        
        # Критерии качества
        length_score = self._score_length(text)
        word_score = self._score_words(text)
        garbage_score = self._score_garbage_ratio(text)
        cyrillic_score = self._score_cyrillic_ratio(text)
        structure_score = self._score_structure(text)
        
        # Если есть референс, добавляем метрику схожести
        similarity_score = 0.0
        if reference:
            similarity_score = self._score_similarity(text, reference)
        
        # Взвешенная оценка
        if reference:
            quality = (
                length_score * 0.15 +
                word_score * 0.15 +
                garbage_score * 0.20 +
                cyrillic_score * 0.15 +
                structure_score * 0.15 +
                similarity_score * 0.20
            )
        else:
            quality = (
                length_score * 0.20 +
                word_score * 0.20 +
                garbage_score * 0.25 +
                cyrillic_score * 0.20 +
                structure_score * 0.15
            )
        
        return min(1.0, max(0.0, quality))
    
    def _score_length(self, text: str) -> float:
        """Оценка по длине текста."""
        length = len(text)
        # Нормализуем: 500+ символов = 1.0, 0 = 0.0
        return min(1.0, length / 500.0)
    
    def _score_words(self, text: str) -> float:
        """Оценка по количеству слов."""
        words = text.split()
        word_count = len([w for w in words if len(w) >= self.min_word_length])
        # Нормализуем: 50+ слов = 1.0
        return min(1.0, word_count / 50.0)
    
    def _score_garbage_ratio(self, text: str) -> float:
        """Оценка по соотношению мусора к нормальному тексту."""
        words = text.split()
        if not words:
            return 0.0
        
        garbage_count = 0
        for word in words:
            for pattern in self.garbage_patterns:
                if re.match(pattern, word):
                    garbage_count += 1
                    break
        
        garbage_ratio = garbage_count / len(words)
        # Инвертируем: меньше мусора = выше оценка
        return 1.0 - min(1.0, garbage_ratio * 2.0)
    
    def _score_cyrillic_ratio(self, text: str) -> float:
        """Оценка по соотношению кириллицы (для русских документов)."""
        if not text:
            return 0.0
        
        cyrillic_count = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c in 'ёЁ')
        total_letters = sum(1 for c in text if c.isalpha())
        
        if total_letters == 0:
            return 0.5  # Нейтральная оценка если нет букв
        
        cyrillic_ratio = cyrillic_count / total_letters
        
        # Для русских документов ожидаем > 70% кириллицы
        if cyrillic_ratio > 0.7:
            return 1.0
        elif cyrillic_ratio > 0.5:
            return 0.7
        elif cyrillic_ratio > 0.3:
            return 0.5
        else:
            return 0.3
    
    def _score_structure(self, text: str) -> float:
        """Оценка по структуре текста (наличие предложений, абзацев)."""
        # Проверяем наличие знаков препинания
        has_punctuation = bool(re.search(r'[.,;:!?]', text))
        
        # Проверяем наличие чисел (часто встречаются в документах)
        has_numbers = bool(re.search(r'\d+', text))
        
        # Проверяем наличие заглавных букв (начало предложений)
        has_capitals = bool(re.search(r'[А-ЯЁA-Z]', text))
        
        # Проверяем наличие пробелов (разделение слов)
        has_spaces = ' ' in text
        
        score = 0.0
        if has_punctuation:
            score += 0.3
        if has_numbers:
            score += 0.2
        if has_capitals:
            score += 0.2
        if has_spaces:
            score += 0.3
        
        return score
    
    def _score_similarity(self, text: str, reference: str) -> float:
        """Оценка схожести с референсным текстом."""
        if not reference:
            return 0.0
        
        # Нормализуем тексты для сравнения
        text_norm = self._normalize_for_comparison(text)
        ref_norm = self._normalize_for_comparison(reference)
        
        # Используем SequenceMatcher для оценки схожести
        similarity = difflib.SequenceMatcher(None, text_norm, ref_norm).ratio()
        
        return similarity
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Нормализует текст для сравнения."""
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        # Приводим к нижнему регистру
        text = text.lower()
        # Убираем знаки препинания для более мягкого сравнения
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()


class OCRResultCombiner:
    """Класс для комбинирования результатов OCR от разных методов."""
    
    def __init__(self):
        self.quality_estimator = OCRQualityEstimator()
    
    def combine_results(self, results: List[Tuple[str, float]], reference: str = "") -> str:
        """
        Комбинирует результаты OCR от разных методов предобработки.
        
        Args:
            results: Список кортежей (текст, confidence)
            reference: Референсный текст (если есть)
            
        Returns:
            Комбинированный текст
        """
        if not results:
            return ""
        
        # Фильтруем пустые результаты
        valid_results = [(text, conf) for text, conf in results if text and text.strip()]
        
        if not valid_results:
            return ""
        
        # Если только один результат, возвращаем его
        if len(valid_results) == 1:
            return valid_results[0][0]
        
        # Оцениваем качество каждого результата
        scored_results = []
        for text, confidence in valid_results:
            quality = self.quality_estimator.estimate_text_quality(text, reference)
            # Комбинируем confidence и quality
            combined_score = (confidence * 0.6 + quality * 0.4)
            scored_results.append((text, combined_score, confidence, quality))
        
        # Сортируем по комбинированному score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Используем взвешенное голосование по строкам
        combined_text = self._weighted_voting(scored_results)
        
        return combined_text
    
    def _weighted_voting(self, scored_results: List[Tuple[str, float, float, float]]) -> str:
        """
        Взвешенное голосование по строкам текста.
        """
        # Группируем строки по нормализованному виду
        line_votes: Dict[str, Dict] = {}
        
        for text, combined_score, confidence, quality in scored_results:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue
                
                # Нормализуем строку для группировки
                normalized = self._normalize_line(line)
                
                if normalized not in line_votes:
                    line_votes[normalized] = {
                        'original': line,
                        'votes': 0,
                        'weighted_score': 0.0,
                        'confidence_sum': 0.0
                    }
                
                line_votes[normalized]['votes'] += 1
                line_votes[normalized]['weighted_score'] += combined_score
                line_votes[normalized]['confidence_sum'] += confidence
        
        # Выбираем строки с достаточным количеством голосов или высокой уверенностью
        final_lines = []
        for normalized, data in line_votes.items():
            avg_confidence = data['confidence_sum'] / data['votes']
            avg_score = data['weighted_score'] / data['votes']
            
            # Критерии включения:
            # - Хотя бы 2 голоса ИЛИ
            # - Высокая уверенность (>0.6) ИЛИ
            # - Высокий комбинированный score (>0.7)
            if (data['votes'] >= 2 or 
                avg_confidence > 0.6 or 
                avg_score > 0.7):
                final_lines.append((data['original'], avg_score))
        
        # Сортируем по score
        final_lines.sort(key=lambda x: x[1], reverse=True)
        
        return '\n'.join(line[0] for line in final_lines)
    
    def _normalize_line(self, line: str) -> str:
        """Нормализует строку для сравнения."""
        # Убираем лишние пробелы
        line = re.sub(r'\s+', ' ', line)
        # Приводим к нижнему регистру
        line = line.lower()
        # Убираем знаки препинания на концах
        line = line.strip('.,;:!?')
        
        return line.strip()

