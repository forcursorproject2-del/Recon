import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from core_parser.config_manager.config_loader import ConfigManager
try:
    from sentence_transformers import SentenceTransformer
    import os
    from pathlib import Path
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logging.warning("sentence-transformers not available, BERT classifier disabled.")

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    rule_score: float
    ml_score: float
    bert_score: float
    explanation: str

class DocumentClassifier:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.get_signatures()
        self.mode = config_manager.get_classifier_mode()
        self.ml_pipeline = None
        self.bert_clf = None
        if config_manager.use_ml():
            self.ml_pipeline = self._init_ml_pipeline()
            logger.info("ML pipeline initialized.")
        if config_manager.use_bert() and BERT_AVAILABLE:
            try:
                model_path = str(Path.home() / ".core_parser" / "rubert-tiny2")
                if os.path.exists(model_path):
                    self.bert_clf = SentenceTransformer(model_path)
                    logger.info(f"Loaded cached BERT model from {model_path}")
                else:
                    self.bert_clf = SentenceTransformer('sergeyzh/rubert-tiny2-ru-go-emotions')
                    self.bert_clf.save(model_path)
                    logger.info(f"Downloaded and cached BERT model to {model_path}")
            except Exception as e:
                logger.warning(f"SentenceTransformer loading failed: {e}")
                self.bert_clf = None
        self.doc_types = list(self.config.keys())
        logger.info(f"Classifier initialized in mode: {self.mode}")

    def _normalize_text(self, text: str) -> str:
        """Нормализация текста для русского языка. Использует централизованный нормализатор."""
        from core_parser.utils.text_normalizer import TextNormalizer
        return TextNormalizer.normalize_for_classification(text)

    def _init_ml_pipeline(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
            ('clf', LogisticRegression(random_state=42))
        ])

    def classify_document(self, text: str, structure: Dict[str, Any]) -> ClassificationResult:
        logger.debug(f"Классификация документа: длина текста {len(text)} символов")
        
        # Если текст слишком короткий, пробуем классифицировать по имени файла
        if len(text.strip()) < 10:
            filename = structure.get('filename', '') or structure.get('file_path', '')
            if filename:
                filename_lower = filename.lower()
                # Проверяем ключевые слова в имени файла
                filename_scores = {}
                for doc_type, sig in self.config.items():
                    score = 0
                    keywords = sig.get('keywords', [])
                    patterns = sig.get('patterns', [])
                    exclude = sig.get('exclude', [])
                    
                    # Проверяем ключевые слова
                    for kw in keywords:
                        if kw.lower() in filename_lower:
                            score += 2
                    
                    # Проверяем паттерны
                    for pat in patterns:
                        if re.search(pat, filename_lower, re.IGNORECASE):
                            score += 3
                    
                    # Проверяем исключения
                    for ex in exclude:
                        if ex.lower() in filename_lower:
                            score -= 2
                    
                    if score > 0:
                        filename_scores[doc_type] = score
                
                if filename_scores:
                    best_type = max(filename_scores, key=filename_scores.get)
                    best_score = filename_scores[best_type]
                    confidence = min(0.9, 0.5 + (best_score / 10))
                    logger.debug(f"Документ классифицирован по имени файла: {best_type} (score: {best_score}, confidence: {confidence})")
                    return ClassificationResult(
                        doc_type=best_type,
                        confidence=confidence,
                        rule_score=confidence,
                        ml_score=0.0,
                        bert_score=0.0,
                        explanation="filename_based"
                    )
        
        if self.mode == "rules_only":
            rule_result = self._rule_based_classification(text)
            return ClassificationResult(
                doc_type=rule_result.doc_type,
                confidence=rule_result.confidence,
                rule_score=rule_result.confidence,
                ml_score=0.0,
                bert_score=0.0,
                explanation="rules_only_mode"
            )
        else:
            rule_result = self._rule_based_classification(text)
            logger.debug(f"Rule-based результат: {rule_result.doc_type} с уверенностью {rule_result.confidence}")
            ml_result = self._ml_classification(text)
            logger.debug(f"ML результат: {ml_result.doc_type} с уверенностью {ml_result.confidence}")
            bert_result = self._bert_classification(text)
            logger.debug(f"BERT результат: {bert_result.doc_type} с уверенностью {bert_result.confidence}")
            # Ensemble: weights [0.3, 0.3, 0.4]
            weights = [0.3, 0.3, 0.4]
            scores = [rule_result.confidence, ml_result.confidence, bert_result.confidence]
            # Increase confidence by 30-50% if BERT confidence > 0.5
            if bert_result.confidence > 0.5:
                scores[2] = min(1.0, scores[2] * 1.5)
            final_confidence = sum(w * s for w, s in zip(weights, scores))
            # Choose doc_type with highest weighted score
            doc_types = [rule_result.doc_type, ml_result.doc_type, bert_result.doc_type]
            best_idx = scores.index(max(scores))
            final_doc_type = doc_types[best_idx]
            logger.debug(f"Финальный результат классификации: {final_doc_type} с уверенностью {final_confidence}")
            return ClassificationResult(final_doc_type, final_confidence, rule_result.confidence, ml_result.confidence, bert_result.confidence, "ensemble_mode")

    def _rule_based_classification(self, text: str) -> ClassificationResult:
        # Нормализация текста с помощью централизованного нормализатора
        from core_parser.utils.text_normalizer import TextNormalizer
        normalized_text = TextNormalizer.normalize_for_classification(text)
        scores = {}
        for doc_type, sig in self.config.items():
            score = 0
            keywords = sig.get('keywords', [])
            patterns = sig.get('patterns', [])
            exclude = sig.get('exclude', [])
            for kw in keywords:
                if kw.lower() in normalized_text:
                    score += 1
            for pat in patterns:
                if re.search(pat, normalized_text, re.IGNORECASE):
                    score += 2
            for ex in exclude:
                if ex.lower() in normalized_text:
                    score -= 1
            
            # Повышаем приоритет для reconciliation_act, если найдены специфичные ключевые слова
            if doc_type == 'reconciliation_act':
                # Проверяем варианты с ошибками OCR
                reconciliation_keywords = [
                    'акт сверки', 'акт сврки', 'акт сверк', 'сверка взаиморасчетов', 
                    'сверка взаиморасчётов', 'заимных расчетов', 'взаимных расчетов',
                    'акт сверки взаиморасчетов', 'акт сверки взаиморасчётов'
                ]
                if any(kw in normalized_text for kw in reconciliation_keywords):
                    score += 5  # Большой бонус за специфичные термины
                
                # Финансовые термины акта сверки
                financial_keywords = ['сальдо', 'дебет', 'кредит', 'обороты', 'начальное сальдо', 
                                     'конечное сальдо', 'входящее сальдо', 'исходящее сальдо']
                if any(kw in normalized_text for kw in financial_keywords):
                    score += 3  # Бонус за финансовые термины
                
                # Платежные поручения в актах сверки
                if 'платежное поручение' in normalized_text or 'латежное поручение' in normalized_text:
                    score += 2
                
                # Нижеподписавшиеся
                if 'нижеподписавшиеся' in normalized_text or 'нижеподлисавшися' in normalized_text:
                    score += 1
            
            scores[doc_type] = max(0, score)
        
        # Если есть несколько типов с одинаковым счетом, приоритет reconciliation_act
        max_score = max(scores.values()) if scores else 0
        best_types = [dt for dt, sc in scores.items() if sc == max_score]
        
        # Специальная логика для актов сверки - повышаем приоритет если есть ключевые слова
        reconciliation_score = scores.get('reconciliation_act', 0)
        if reconciliation_score > 0:
            # Если есть хотя бы слабые признаки акта сверки, повышаем приоритет
            if any(kw in normalized_text for kw in ['акт', 'сверк', 'заимных', 'взаимных', 'сальдо', 'дебет', 'кредит']):
                reconciliation_score += 2
        
        # Если reconciliation_act имеет высокий счет или есть в лучших типах, выбираем его
        if len(best_types) > 1 and 'reconciliation_act' in best_types:
            best_type = 'reconciliation_act'
        elif reconciliation_score >= max_score and reconciliation_score > 3:
            # Если счет акта сверки достаточно высокий, выбираем его
            best_type = 'reconciliation_act'
        else:
            best_type = max(scores, key=scores.get) if scores else 'unknown'
        score = scores[best_type] if best_type in scores else 0
        # Исправленный расчет confidence
        if score >= 3:
            confidence = 1.0
        elif score >= 2:
            confidence = 0.95
        elif score >= 1:
            confidence = 0.80
        else:
            confidence = 0.0
        return ClassificationResult(best_type, confidence, confidence, 0.0, 0.0, "")

    def _ml_classification(self, text: str) -> ClassificationResult:
        if not self.ml_pipeline or not hasattr(self.ml_pipeline.named_steps['clf'], 'classes_'):
            return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
        try:
            if not text or not text.strip():
                logger.warning("Пустой текст для ML-классификации")
                return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
            
            proba = self.ml_pipeline.predict_proba([text])[0]
            best_idx = proba.argmax()
            best_class = self.ml_pipeline.named_steps['clf'].classes_[best_idx]
            confidence = proba[best_idx]
            return ClassificationResult(best_class, confidence, 0.0, confidence, 0.0, "")
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")

    def _bert_classification(self, text: str) -> ClassificationResult:
        if not self.bert_clf:
            return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
        try:
            # Убедимся, что текст не пустой
            if not text or not text.strip():
                logger.warning("Пустой текст для BERT-классификации")
                return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
            
            # Ограничиваем длину текста для BERT (максимум 512 токенов)
            processed_text = text[:512].strip()
            if not processed_text:
                logger.warning("После обрезки текст оказался пустым для BERT-классификации")
                return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
            
            # Нормализуем текст перед BERT-обработкой
            processed_text = self._normalize_text(processed_text)
            
            # Получаем эмбеддинг текста документа
            doc_embedding = self.bert_clf.encode(processed_text)
            
            # Создаем эталонные эмбеддинги для каждого типа документа на основе ключевых слов
            best_similarity = 0.0
            best_doc_type = 'unknown'
            
            for doc_type in self.doc_types:
                signature = self.config.get(doc_type, {})
                keywords = signature.get('keywords', [])
                patterns = signature.get('patterns', [])
                
                # Создаем эталонный текст для типа документа на основе ключевых слов
                reference_text = ' '.join(keywords + patterns)
                if reference_text:
                    ref_embedding = self.bert_clf.encode(self._normalize_text(reference_text))
                    
                    # Вычисляем косинусное сходство
                    similarity = self._cosine_similarity(doc_embedding, ref_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_doc_type = doc_type
            
            # Нормализуем схожесть в диапазон [0, 1]
            confidence = min(1.0, max(0.0, best_similarity))
            
            return ClassificationResult(best_doc_type, confidence, 0.0, 0.0, confidence, "")
        except Exception as e:
            logger.warning(f"BERT classification failed: {e}")
            return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Вычисляет косинусное сходство между двумя векторами"""
        import numpy as np
        
        # Преобразуем в numpy массивы, если они еще не таковы
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
            
        # Нормализуем векторы
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Вычисляем косинусное сходство
        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
        
        # Возвращаем значение в диапазоне [0, 1]
        return (cosine_sim + 1) / 2  # Приводим к диапазону [0, 1] из [-1, 1]

    def train_on_labeled_data(self, labeled_texts: List[Tuple[str, str]]):
        texts, labels = zip(*labeled_texts)
        self.ml_pipeline.fit(texts, labels)
        joblib.dump(self.ml_pipeline, 'model.pkl')
        logger.info("Model trained and saved.")

class BatchClassifier:
    def __init__(self, classifier: DocumentClassifier):
        self.classifier = classifier

    def classify_batch(self, documents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        logger.debug(f"Классификация батча из {len(documents)} документов")
        results = {}
        statistics = {'classified': 0, 'uncertain': 0, 'types': {}}
        for filename, data in documents.items():
            logger.debug(f"Классификация документа: {filename}")
            # Убеждаемся, что filename есть в структуре для классификации по имени файла
            if 'filename' not in data:
                data['filename'] = filename
            if 'file_path' not in data:
                data['file_path'] = filename
            result = self.classifier.classify_document(data.get('full_text', ''), data)
            results[filename] = result
            if result.confidence > 0.5:
                statistics['classified'] += 1
            else:
                statistics['uncertain'] += 1
            statistics['types'][result.doc_type] = statistics['types'].get(result.doc_type, 0) + 1
        logger.debug(f"Классификация батча завершена: {statistics}")
        return {'results': results, 'statistics': statistics}
