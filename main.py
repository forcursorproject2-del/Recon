# main.py
import eel
import json
import shutil
from pathlib import Path
import sys
import logging

# === Путь для .exe ===
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# === Логирование ===
from core_parser.logger import setup_logging
import logging
import re

# Включаем расширенное логирование
setup_logging('logs/debug.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Включаем максимальный уровень логирования для нужных модулей
logging.getLogger('core_parser.pdf_extractor.pdf_reader').setLevel(logging.DEBUG)
logging.getLogger('core_parser.classifier.document_classifier').setLevel(logging.DEBUG)
logging.getLogger('core_parser.batch_processor.pipeline').setLevel(logging.DEBUG)

from core_parser.config_manager.config_loader import ConfigManager
from core_parser.batch_processor.pipeline import BatchProcessingPipeline
from core_parser.classifier.document_classifier import DocumentClassifier

from pathlib import Path
import json
import shutil
import os
import eel

config = ConfigManager('core_parser/config.yaml')
logger.info(f"Classifier mode: {config.get_classifier_mode()}, ML: {config.use_ml()}, BERT: {config.use_bert()}")

logger.info("Первый запуск: загрузка моделей PaddleOCR и RuBERT-tiny2 (~450 МБ)")

pipeline = BatchProcessingPipeline(config, use_ocr=True, max_workers=4)

def check_text_issues(text: str):
    keywords = [r"акт сверки", r"взаиморасчётов", r"сальдо", r"по состоянию на", r"2025 г."]
    found_keywords = [kw for kw in keywords if re.search(kw, text, re.IGNORECASE)]
    logger.info(f"Найденные ключевые слова/выражения: {found_keywords}")

    if 'ё' in text or 'Ё' in text:
        logger.info("Текст содержит букву 'ё'. Проверяем корректность замены ё на е в классификаторе.")
    else:
        logger.info("Буква 'ё' не найдена в тексте.")

    ocr_error_patterns = {
        "Акт сверки": ["Акт сверкии", "Акт сверкы"],
        "расчётов": ["расчетов", "расчетоов", "расчетооы"],
        "сальдо": ["сальдоо", "салдо"],
    }

    ocr_errors_found = []
    text_lower = text.lower()
    for correct, errors in ocr_error_patterns.items():
        for err in errors:
            if err.lower() in text_lower:
                ocr_errors_found.append(f"Потенциальная OCR-ошибка: '{err}' вместо '{correct}'")

    if ocr_errors_found:
        for err in ocr_errors_found:
            logger.warning(err)
    else:
        logger.info("Не обнаружено типичных OCR-ошибок в тексте.")


if __name__ == "__main__":
    logger.info("Запуск приложения")
    logger.debug("Создание необходимых папок")
    for folder in ["incoming", "processed", "results", "logs"]:
        Path(folder).mkdir(exist_ok=True)
        logger.debug(f"Папка {folder} проверена/создана")

    def run_reconciliation_act_test():
        config_manager = ConfigManager()
        config = config_manager.config
        extractor = pipeline.pdf_processor.extractor
        pdf_path = Path('incoming') / 'CCF_003705.pdf'
        extraction_result = extractor.extract_text_with_structure(str(pdf_path))
        full_text = extraction_result.get('full_text', '')
        logger.info(f"Полный извлечённый текст ({len(full_text)} символов):\n{full_text}")

        ocr_used = pipeline.pdf_processor.use_ocr and not any(page['text'].strip() for page in extraction_result.get('pages', []))
        logger.info(f"OCR включен: {pipeline.pdf_processor.use_ocr}")
        logger.info(f"OCR фактически использован (нет текста из pdfplumber): {ocr_used}")
        logger.info(f"Количество символов извлечено: {len(full_text)}")

        classifier = DocumentClassifier(config_manager)
        classification_result = classifier.classify_document(full_text, extraction_result)

        normalized_text = full_text.replace('ё', 'е').replace('Ё', 'Е')
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        scores = {}
        for doc_type, sig in classifier.config.items():
            score = 0
            for kw in sig.get('keywords', []):
                if kw.lower() in normalized_text.lower():
                    score += 1
            for pat in sig.get('patterns', []):
                if re.search(pat, normalized_text, re.IGNORECASE):
                    score += 2
            for ex in sig.get('exclude', []):
                if ex.lower() in normalized_text.lower():
                    score -= 1
            scores[doc_type] = max(0, score)
        logger.info(f"Rule-based классификация scores по типам документов: {scores}")

        logger.info(f"Финальный doc_type: {classification_result.doc_type}, confidence: {classification_result.confidence}")

        if classification_result.confidence < 0.5:
            snippet_start = full_text[:1000]
            snippet_end = full_text[-500:]
            logger.warning(f"Низкий уровень доверия (<0.5). Первые 1000 символов:\n{snippet_start}")
            logger.warning(f"Последние 500 символов:\n{snippet_end}")

        check_text_issues(full_text)

    run_reconciliation_act_test()

    logger.debug("Инициализация Eel")
    eel.init('web', allowed_extensions=['.js', '.html', '.css'])
    logger.info("Eel инициализирован, запуск веб-интерфейса")
    eel.start('index.html', size=(1600, 1000), port=0)
