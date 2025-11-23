import logging
import os
from pathlib import Path

def setup_logging(log_file: str = 'debug.log', level: int = logging.DEBUG):
    """
    Настраивает детальное логирование для записи в файл и консоль.
    """
    # Создаем директорию для логов, если не существует
    log_dir = Path(log_file).parent
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Настройка детального формата
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

    # Хендлер для файла
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Хендлер для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Консоль менее детальная

    # Настройка корневого логгера
    logging.basicConfig(
        level=level,
        handlers=[file_handler, console_handler]
    )

    # Отключаем логи от библиотек, если не нужно
    logging.getLogger('pdfplumber').setLevel(logging.WARNING)
    logging.getLogger('fitz').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('spacy').setLevel(logging.WARNING)
    logging.getLogger('camelot').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
