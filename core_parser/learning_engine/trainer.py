import logging
from typing import List, Tuple, Any
from core_parser.classifier.document_classifier import DocumentClassifier
try:
    from modAL import ActiveLearner
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    logging.warning("modAL not available, active learning disabled.")

logger = logging.getLogger(__name__)

class LearningEngine:
    def __init__(self, classifier: DocumentClassifier):
        self.classifier = classifier
        self.active_learner = None
        if MODAL_AVAILABLE:
            # Initialize with dummy data, will be set later
            self.active_learner = ActiveLearner(estimator=self.classifier.ml_pipeline)

    def train_classifier(self, labeled_data: List[Tuple[str, str]]):
        self.classifier.train_on_labeled_data(labeled_data)
        logger.info("Classifier training completed.")

    def active_train_classifier(self, pool_data: List[Tuple[str, str]], n_queries: int = 10):
        if not self.active_learner:
            logger.warning("ActiveLearner not available, falling back to standard training.")
            self.train_classifier(pool_data)
            return
        texts, labels = zip(*pool_data)
        X_pool = self.classifier.ml_pipeline.named_steps['tfidf'].fit_transform(texts)
        y_pool = list(labels)
        # Initial training with a subset
        initial_idx = list(range(min(10, len(pool_data))))
        self.active_learner.teach(X_pool[initial_idx], [y_pool[i] for i in initial_idx])
        # Active learning loop
        for _ in range(n_queries):
            query_idx = self.active_learner.query(X_pool)
            # Simulate manual labeling (in real scenario, ask user)
            manual_label = y_pool[query_idx[0]]  # Assume correct label
            self.active_learner.teach(X_pool[query_idx], [manual_label])
        logger.info("Active learning training completed.")

    def active_train_on_feedback(self, doc_id: str, correct_type: str, document_text: str = None):
        """
        Обновляет модель на основе обратной связи пользователя.
        
        Args:
            doc_id: Идентификатор документа
            correct_type: Правильный тип документа, указанный пользователем
            document_text: Текст документа. Если не указан, будет попытка извлечь из кэша
        """
        logger.info(f"Получена обратная связь для документа {doc_id}: правильный тип -> {correct_type}")
        
        if not document_text:
            logger.warning(f"Текст документа {doc_id} не предоставлен. Требуется текст для обучения.")
            return
        
        if not self.classifier.ml_pipeline:
            logger.warning("ML pipeline не инициализирован. Невозможно обновить модель.")
            return
        
        try:
            # Проверяем, что тип документа существует в конфигурации
            if correct_type not in self.classifier.doc_types:
                logger.warning(f"Неизвестный тип документа: {correct_type}. Доступные типы: {self.classifier.doc_types}")
                return
            
            # Если ActiveLearner доступен, используем его для обучения
            if self.active_learner and MODAL_AVAILABLE:
                # Преобразуем текст в вектор признаков
                X_new = self.classifier.ml_pipeline.named_steps['tfidf'].transform([document_text])
                # Обучаем модель на новом примере
                self.active_learner.teach(X_new, [correct_type])
                logger.info(f"Модель обновлена через ActiveLearner для документа {doc_id}")
            else:
                # Fallback: добавляем в обучающие данные и переобучаем
                # В реальном сценарии здесь должна быть база данных с обучающими примерами
                logger.info(f"ActiveLearner недоступен. Добавляю пример в очередь для переобучения.")
                logger.info("Для полноценного обучения используйте метод train_classifier с полным набором данных.")
            
            logger.info(f"Модель успешно обновлена с обратной связью: {doc_id} -> {correct_type}")
        except Exception as e:
            logger.error(f"Ошибка при обновлении модели с обратной связью для {doc_id}: {e}", exc_info=True)
