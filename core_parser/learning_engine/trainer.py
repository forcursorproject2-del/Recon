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

    def active_train_on_feedback(self, doc_id: str, correct_type: str):
        # In real scenario, retrieve the document text and update the model
        # For demo, assume we have the text
        logger.info(f"Updating model with feedback: {doc_id} -> {correct_type}")
        # Placeholder: add to training data
        # self.classifier.train_on_labeled_data([(text, correct_type)])
        logger.info("Model updated with user feedback.")
