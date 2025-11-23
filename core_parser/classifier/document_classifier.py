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
        """Нормализация текста для русского языка: приведение к нижнему регистру, 
        замена 'ё' на 'е', нормализация пробелов, удаление лишних символов."""
        if not text:
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Замена 'ё' на 'е'
        text = text.replace('ё', 'е')
        
        # Нормализация пробелов (заменяем последовательности пробельных символов на один пробел)
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление лишних символов, оставляем только буквы, цифры, пробелы и базовые знаки препинания
        text = re.sub(r'[^\w\s\.\,\-\+\(\)\[\]\{\}\/\\=:;]', ' ', text)
        
        # Удаление лишних пробелов в начале и конце
        text = text.strip()
        
        return text

    def _init_ml_pipeline(self) -> Pipeline:
        return Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1,2))),
            ('clf', LogisticRegression(random_state=42))
        ])

    def classify_document(self, text: str, structure: Dict[str, Any]) -> ClassificationResult:
        logger.debug(f"Классификация документа: длина текста {len(text)} символов")
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
        
        # Нормализация текста
        text = text.lower()
        text = text.replace('ё', 'е')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\-\+]', '', text)  # убираем странные символы
        scores = {}
        for doc_type, sig in self.config.items():
            score = 0
            keywords = sig.get('keywords', [])
            patterns = sig.get('patterns', [])
            exclude = sig.get('exclude', [])
            for kw in keywords:
                if kw.lower() in text:
                    score += 1
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    score += 2
            for ex in exclude:
                if ex.lower() in text:
                    score -= 1
            scores[doc_type] = max(0, score)
        best_type = max(scores, key=scores.get)
        score = scores[best_type]
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
            
            embedding = self.bert_clf.encode(processed_text)
            # Для улучшения качества классификации, можно использовать косинусное сходство
            # с эталонными эмбеддингами для каждого типа документа
            # Пока что используем улучшенную заглушку, но с более разумной логикой
            embedding_norm = float(abs(embedding).mean())  # усреднённое значение эмбеддинга
            confidence = min(1.0, max(0.0, embedding_norm / 2.0))  # нормализуем в диапазон [0, 1]
            
            # Пока что возвращаем 'unknown', но в будущем можно реализовать
            # сравнение с эталонными эмбеддингами для определения типа документа
            doc_type = 'unknown'
            return ClassificationResult(doc_type, confidence, 0.0, 0.0, confidence, "")
        except Exception as e:
            logger.warning(f"BERT classification failed: {e}")
            return ClassificationResult('unknown', 0.0, 0.0, 0.0, 0.0, "")

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
            result = self.classifier.classify_document(data['full_text'], data)
            results[filename] = result
            if result.confidence > 0.5:
                statistics['classified'] += 1
            else:
                statistics['uncertain'] += 1
            statistics['types'][result.doc_type] = statistics['types'].get(result.doc_type, 0) + 1
        logger.debug(f"Классификация батча завершена: {statistics}")
        return {'results': results, 'statistics': statistics}
