# main.py
import eel
import json
import shutil
from pathlib import Path
import sys
import logging
import os
import re
import time
from typing import Any, List, Dict

# === Путь для .exe ===
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path

# === Логирование ===
from core_parser.logger import setup_logging

# Включаем расширенное логирование
setup_logging('logs/debug.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Включаем максимальный уровень логирования для нужных модулей
logging.getLogger('core_parser.pdf_extractor.pdf_reader').setLevel(logging.DEBUG)
logging.getLogger('core_parser.classifier.document_classifier').setLevel(logging.DEBUG)
logging.getLogger('core_parser.batch_processor.pipeline').setLevel(logging.DEBUG)

from core_parser.config_manager.config_loader import ConfigManager
from core_parser.batch_processor.pipeline import BatchProcessingPipeline
from core_parser.config_manager.preferred_fields import get_preferred_fields, is_preferred_field
from core_parser.utils.processing_history import ProcessingHistory

config = ConfigManager('core_parser/config.yaml')
logger.info(f"Classifier mode: {config.get_classifier_mode()}, ML: {config.use_ml()}, BERT: {config.use_bert()}")

logger.info("Первый запуск: загрузка моделей PaddleOCR и RuBERT-tiny2 (~450 МБ)")

pipeline = BatchProcessingPipeline(config, use_ocr=True, max_workers=4)
history = ProcessingHistory()

# === Вспомогательные функции ===

def _is_table_header(column_name: str, column_value: Any, table: List[Dict]) -> bool:
    """
    Определяет, является ли колонка заголовком таблицы или значением данных.
    
    Заголовки обычно:
    - Короткие (до 50 символов)
    - Не содержат длинных текстовых описаний
    - Не содержат реквизитов (БИК, счета, названия банков)
    - Не содержат только цифры или даты
    - Повторяются в нескольких строках как ключи
    """
    if not column_name or not isinstance(column_name, str):
        return False
    
    # Слишком длинные значения - скорее всего данные, а не заголовки
    if len(column_name) > 50:
        return False
    
    # Значения, содержащие реквизиты банков - это данные
    bank_indicators = ['банк', 'бik', 'бик', 'сч.', 'счет', 'р/с', 'к/с', 'получатель', 'отправитель']
    column_lower = column_name.lower()
    if any(indicator in column_lower for indicator in bank_indicators):
        # Проверяем, не является ли это просто меткой "БИК" или "Счет"
        if column_lower in ['бик', 'бik', 'сч.', 'счет', 'р/с', 'к/с']:
            return True  # Это заголовок
        # Если содержит длинный текст с реквизитами - это данные
        if len(column_name) > 15:
            return False
    
    # Значения, содержащие только цифры - скорее всего данные
    if re.match(r'^\d+$', str(column_name).strip()):
        return False
    
    # Значения, содержащие длинные текстовые описания - скорее всего данные
    # Например: "КБ «БАНК-премиум» г. Москва Банк получателя"
    if '«' in column_name or '»' in column_name:
        if len(column_name) > 20:
            return False
    
    # Если значение содержит только None или пустую строку - пропускаем
    if column_value is None or (isinstance(column_value, str) and not column_value.strip()):
        # Но если это ключ, который используется в других строках - это заголовок
        if isinstance(table, list) and len(table) > 1:
            # Проверяем, используется ли этот ключ в других строках
            used_as_key = sum(1 for row in table[1:6] if isinstance(row, dict) and column_name in row)
            if used_as_key > 0:
                return True
        return False
    
    # Стандартные заголовки таблиц (короткие метки)
    common_headers = ['№', 'номер', 'наименование', 'название', 'ед.', 'ед', 'кол-во', 'количество', 
                     'цена', 'сумма', 'стоимость', 'итого', 'всего', 'ндс', 'дата', 'коммент', 'комментарий',
                     'результат', 'исследование', 'ед. изм.', 'цена за ед.', 'стоимость товаров']
    if column_lower in common_headers or any(header in column_lower for header in common_headers):
        return True
    
    # Если ключ используется в нескольких строках как ключ - это заголовок
    if isinstance(table, list) and len(table) > 1:
        used_as_key = sum(1 for row in table[1:min(10, len(table))] if isinstance(row, dict) and column_name in row)
        if used_as_key >= 2:  # Используется в 2+ строках
            return True
    
    return True  # По умолчанию считаем заголовком, если не доказано обратное

# === Eel функции для веб-интерфейса ===

@eel.expose
def get_folder_counts():
    """Возвращает количество файлов в папках incoming, processed, results."""
    incoming_path = Path("incoming")
    processed_path = Path("processed")
    results_path = Path("results")
    
    incoming_count = len(list(incoming_path.glob("*.pdf"))) if incoming_path.exists() else 0
    processed_count = len(list(processed_path.glob("*.pdf"))) if processed_path.exists() else 0
    results_count = len(list(results_path.glob("*.json"))) if results_path.exists() else 0
    
    return {
        'incoming': incoming_count,
        'processed': processed_count,
        'results': results_count
    }

@eel.expose
def run_parsing(options_json):
    """Запускает обработку документов из папки incoming."""
    try:
        incoming_path = Path("incoming")
        if not incoming_path.exists():
            return {'status': 'error', 'message': 'Папка incoming не найдена'}
        
        results = pipeline.process_folder(str(incoming_path))
        
        # Группируем результаты по типам документов
        document_groups = {}
        documents = results.get('documents', {})
        classifications = results.get('classifications', {})
        
        for filename, data in documents.items():
            # Получаем тип документа из классификации или из данных
            doc_type = 'unknown'
            confidence = 0.0
            
            if filename in classifications:
                # Берем из classifications словаря
                doc_type = classifications[filename].get('doc_type', 'unknown')
                confidence = classifications[filename].get('confidence', 0.0)
            elif 'doc_type' in data:
                # Берем из данных документа
                doc_type = data.get('doc_type', 'unknown')
                confidence = data.get('confidence', 0.0)
            elif 'classification' in data:
                # Берем из вложенной структуры classification
                cls_data = data.get('classification', {})
                doc_type = cls_data.get('doc_type', 'unknown')
                confidence = cls_data.get('confidence', 0.0)
            
            if doc_type not in document_groups:
                document_groups[doc_type] = {
                    'count': 0,
                    'confidence': 0.0,
                    'fields': {},
                    'documents': []
                }
            
            document_groups[doc_type]['count'] += 1
            document_groups[doc_type]['confidence'] += confidence
            
            # Собираем ВСЕ поля из документа
            # 1. Поля из fields
            fields = data.get('fields', {})
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    # Записываем поле, даже если значение None (для статистики)
                    if field_name not in document_groups[doc_type]['fields']:
                        document_groups[doc_type]['fields'][field_name] = {
                            'count': 0,
                            'total': 0,
                            'examples': []
                        }
                    if field_data.get('value') is not None:
                        document_groups[doc_type]['fields'][field_name]['count'] += 1
                        # Сохраняем пример значения (максимум 3 примера)
                        if len(document_groups[doc_type]['fields'][field_name]['examples']) < 3:
                            example_value = str(field_data.get('value', ''))
                            if example_value not in document_groups[doc_type]['fields'][field_name]['examples']:
                                document_groups[doc_type]['fields'][field_name]['examples'].append(example_value)
                    document_groups[doc_type]['fields'][field_name]['total'] += 1
            
            # 2. Поля из таблиц (колонки таблиц)
            # Улучшенная логика: различаем заголовки и значения
            tables = data.get('tables', [])
            for table in tables:
                if isinstance(table, list) and len(table) > 0:
                    # Берем первую строку таблицы как заголовки
                    if isinstance(table[0], dict):
                        # Собираем все значения из первой строки для анализа
                        first_row_values = list(table[0].values())
                        first_row_keys = list(table[0].keys())
                        
                        # Определяем, какие ключи являются заголовками, а какие - значениями
                        # Заголовки обычно: короткие, не содержат длинных текстов, не содержат реквизитов
                        for idx, column_name in enumerate(first_row_keys):
                            column_value = first_row_values[idx] if idx < len(first_row_values) else None
                            
                            # Проверяем, является ли это значение заголовком или данными
                            if not _is_table_header(column_name, column_value, table):
                                continue  # Пропускаем значения, которые не являются заголовками
                            
                            table_field_name = f"table_{column_name}"
                            if table_field_name not in document_groups[doc_type]['fields']:
                                document_groups[doc_type]['fields'][table_field_name] = {
                                    'count': 0,
                                    'total': 0,
                                    'examples': [],
                                    'type': 'table_column'
                                }
                            # Проверяем, есть ли непустые значения в этой колонке
                            has_values = any(row.get(column_name) not in [None, '', 'NaN'] for row in table[:5] if isinstance(row, dict))
                            if has_values:
                                document_groups[doc_type]['fields'][table_field_name]['count'] += 1
                            document_groups[doc_type]['fields'][table_field_name]['total'] += 1
            
            # 3. Поля из table_fields (для счетов - товары и т.д.)
            table_fields = data.get('table_fields', {})
            for field_name, field_value in table_fields.items():
                if field_name not in document_groups[doc_type]['fields']:
                    document_groups[doc_type]['fields'][field_name] = {
                        'count': 0,
                        'total': 0,
                        'examples': [],
                        'type': 'table_data'
                    }
                if field_value:
                    document_groups[doc_type]['fields'][field_name]['count'] += 1
                document_groups[doc_type]['fields'][field_name]['total'] += 1
            
            document_groups[doc_type]['documents'].append(filename)
        
        # Вычисляем среднюю уверенность и преобразуем формат полей для фронтенда
        for doc_type in document_groups:
            if document_groups[doc_type]['count'] > 0:
                document_groups[doc_type]['confidence'] /= document_groups[doc_type]['count']
            
            # Преобразуем формат полей для удобного отображения
            # Если поля уже в простом формате, оставляем как есть
            # Если в расширенном формате (с count/total/examples), преобразуем
            formatted_fields = {}
            for field_name, field_info in document_groups[doc_type]['fields'].items():
                if isinstance(field_info, dict):
                    formatted_fields[field_name] = field_info.get('count', 0)
                else:
                    formatted_fields[field_name] = field_info
            
            # Сортируем поля: сначала избранные, затем остальные
            preferred = get_preferred_fields(doc_type)
            sorted_fields = {}
            
            # Сначала добавляем избранные поля в порядке их приоритета
            for preferred_field in preferred:
                if preferred_field in formatted_fields:
                    sorted_fields[preferred_field] = formatted_fields[preferred_field]
            
            # Затем добавляем остальные поля
            for field_name, field_count in formatted_fields.items():
                if field_name not in sorted_fields:
                    sorted_fields[field_name] = field_count
            
            # Сохраняем отсортированные поля
            document_groups[doc_type]['fields'] = sorted_fields
        
        # Сохраняем результаты для последующего экспорта
        try:
            processing_results_path = Path("results") / "processing_results.json"
            processing_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(processing_results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.debug(f"Результаты обработки сохранены: {processing_results_path}")
        except Exception as save_error:
            logger.error(f"Ошибка при сохранении результатов обработки: {save_error}", exc_info=True)
            # Не прерываем выполнение, но логируем ошибку
        
        return {
            'status': 'success',
            'data': {
                'document_groups': document_groups
            }
        }
    except Exception as e:
        logger.error(f"Ошибка при обработке документов: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def save_mapping(doc_type, mapping_data):
    """Сохраняет маппинг полей для экспорта."""
    try:
        # Сохраняем маппинг в файл
        mappings_path = Path("results") / "mappings.json"
        mappings = {}
        
        if mappings_path.exists():
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        
        mappings[doc_type] = mapping_data
        
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Маппинг сохранен для типа документа: {doc_type}")
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Ошибка при сохранении маппинга: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def submit_feedback(doc_type, correct_type, documents):
    """Принимает обратную связь от пользователя для улучшения модели."""
    try:
        from core_parser.classifier.document_classifier import DocumentClassifier
        from core_parser.learning_engine.trainer import LearningEngine
        
        classifier = DocumentClassifier(config)
        learning_engine = LearningEngine(classifier)
        
        logger.info(f"Получена обратная связь: {doc_type} -> {correct_type} для {len(documents)} документов")
        
        # Сохраняем обратную связь для последующего обучения
        feedback_path = Path("results") / "feedback.json"
        feedback_data = []
        
        if feedback_path.exists():
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        
        for doc in documents:
            feedback_data.append({
                'document': doc,
                'original_type': doc_type,
                'correct_type': correct_type,
                'timestamp': time.time()
            })
        
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return {'status': 'success', 'message': 'Обратная связь сохранена для обучения модели'}
    except Exception as e:
        logger.error(f"Ошибка при обработке обратной связи: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def run_final_export():
    """Выполняет финальный экспорт всех обработанных документов."""
    try:
        from core_parser.utils.excel_exporter import ExcelExporter
        
        mappings_path = Path("results") / "mappings.json"
        if not mappings_path.exists():
            return {'status': 'error', 'message': 'Маппинги не найдены. Настройте экспорт сначала.'}
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        results_path = Path("results") / "processing_results.json"
        if not results_path.exists():
            # Пробуем results.json как fallback
            results_path = Path("results") / "results.json"
        
        if not results_path.exists():
            return {'status': 'error', 'message': 'Результаты обработки не найдены. Выполните обработку сначала.'}
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        logger.info(f"Выполняется финальный экспорт для {len(mappings)} типов документов")
        
        exporter = ExcelExporter()
        count = 0
        exported_files = []
        total_documents = 0
        
        documents = results_data.get('documents', {})
        
        for doc_type, mapping_config in mappings.items():
            # Собираем все документы этого типа
            type_documents = {}
            for filename, doc_data in documents.items():
                # Проверяем тип документа
                doc_doc_type = None
                if 'doc_type' in doc_data:
                    doc_doc_type = doc_data['doc_type']
                elif 'classification' in doc_data:
                    doc_doc_type = doc_data['classification'].get('doc_type')
                
                if doc_doc_type == doc_type:
                    type_documents[filename] = doc_data
            
            if not type_documents:
                logger.warning(f"Не найдено документов типа '{doc_type}' для экспорта")
                continue
            
            # Определяем путь к файлу
            if mapping_config.get('target') == 'existing':
                file_path = Path("results") / mapping_config.get('filename', f'{doc_type}.xlsx')
            else:
                file_path = Path("results") / mapping_config.get('filename', f'{doc_type}.xlsx')
            
            # Экспортируем все документы этого типа в один файл
            exported_count = exporter.export_documents_group(
                doc_type=doc_type,
                documents=type_documents,
                mapping=mapping_config,
                output_path=file_path
            )
            
            if exported_count > 0:
                exported_files.append({
                    'file': str(file_path),
                    'doc_type': doc_type,
                    'count': exported_count
                })
                total_documents += exported_count
                count += 1
                logger.info(f"Экспортировано {exported_count} документов типа '{doc_type}' в {file_path}")
        
        logger.info(f"Экспорт завершен. Обработано {count} типов, {total_documents} документов")
        return {
            'status': 'success',
            'count': total_documents,
            'files': exported_files,
            'message': f'Экспортировано {total_documents} документов в {count} файлов'
        }
    except Exception as e:
        logger.error(f"Ошибка при финальном экспорте: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def export_to_excel():
    """Экспортирует результаты в Excel. Использует run_final_export() для правильного экспорта."""
    logger.warning("export_to_excel() устарела, используйте run_final_export() для экспорта с маппингом")
    return run_final_export()

@eel.expose
def get_processing_history(page=1, per_page=50, filters_json='{}'):
    """Получает историю обработки документов с пагинацией и фильтрами."""
    try:
        import json
        filters = json.loads(filters_json) if filters_json else {}
        result = history.get_history(page=page, per_page=per_page, filters=filters)
        return {'status': 'success', 'data': result}
    except Exception as e:
        logger.error(f"Ошибка при получении истории: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def get_history_statistics():
    """Получает статистику по обработанным документам."""
    try:
        stats = history.get_statistics()
        return {'status': 'success', 'data': stats}
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def open_file(file_path):
    """Открывает файл в системном просмотрщике."""
    try:
        import subprocess
        import platform
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {'status': 'error', 'message': 'Файл не найден'}
        
        system = platform.system()
        if system == 'Windows':
            os.startfile(str(file_path_obj))
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', str(file_path_obj)], check=False)
        else:  # Linux
            subprocess.run(['xdg-open', str(file_path_obj)], check=False)
        
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Ошибка при открытии файла: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def reprocess_document(document_id):
    """Переобрабатывает документ из истории."""
    try:
        # Получаем информацию о документе
        doc = history.get_document_by_id(document_id)
        if not doc:
            return {'status': 'error', 'message': 'Документ не найден'}
        
        original_path = doc.get('original_path')
        
        if not original_path or not Path(original_path).exists():
            return {'status': 'error', 'message': 'Исходный файл не найден'}
        
        # Копируем файл обратно в incoming для переобработки
        incoming_path = Path("incoming") / Path(original_path).name
        incoming_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(original_path, incoming_path)
        
        return {'status': 'success', 'message': 'Документ скопирован в папку incoming для переобработки'}
    except Exception as e:
        logger.error(f"Ошибка при переобработке документа: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def delete_from_history(document_id):
    """Удаляет запись из истории обработки."""
    try:
        success = history.delete_document(document_id)
        if success:
            return {'status': 'success', 'message': 'Запись удалена из истории'}
        else:
            return {'status': 'error', 'message': 'Запись не найдена'}
    except Exception as e:
        logger.error(f"Ошибка при удалении из истории: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def find_duplicates(document_id):
    """Находит возможные дубликаты документа."""
    try:
        duplicates = history.find_duplicates(document_id)
        return {'status': 'success', 'data': duplicates}
    except Exception as e:
        logger.error(f"Ошибка при поиске дубликатов: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def get_contractors_list():
    """Получает список всех контрагентов для фильтра."""
    try:
        contractors = history.get_contractors_list()
        return {'status': 'success', 'data': contractors}
    except Exception as e:
        logger.error(f"Ошибка при получении списка контрагентов: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

@eel.expose
def export_history_to_excel(filters_json='{}'):
    """Экспортирует историю обработки в Excel."""
    try:
        import json
        from core_parser.utils.excel_exporter import ExcelExporter
        import pandas as pd
        
        filters = json.loads(filters_json) if filters_json else {}
        
        # Получаем все записи (без пагинации)
        result = history.get_history(page=1, per_page=10000, filters=filters)
        documents = result['documents']
        
        if not documents:
            return {'status': 'error', 'message': 'Нет данных для экспорта'}
        
        # Преобразуем в DataFrame
        df_data = []
        for doc in documents:
            df_data.append({
                'Исходное имя файла': doc.get('original_filename', ''),
                'Тип документа': doc.get('document_type', ''),
                'Статус обработки': doc.get('status', ''),
                'Контрагент': doc.get('contractor_name', ''),
                'ИНН контрагента': doc.get('contractor_inn', ''),
                'Дата документа': doc.get('document_date', ''),
                'Номер документа': doc.get('document_number', ''),
                'Сумма': doc.get('total_amount', ''),
                'Сумма с НДС': doc.get('amount_with_vat', ''),
                'Сумма без НДС': doc.get('amount_without_vat', ''),
                'Дата обработки': doc.get('processed_at', ''),
                'Уверенность': doc.get('confidence', ''),
                'Ошибка': doc.get('error_message', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Сохраняем в Excel
        output_path = Path("results") / "history_export.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        return {'status': 'success', 'file': str(output_path), 'count': len(documents)}
    except Exception as e:
        logger.error(f"Ошибка при экспорте истории: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    logger.info("Запуск приложения")
    logger.debug("Создание необходимых папок")
    for folder in ["incoming", "processed", "results", "logs"]:
        Path(folder).mkdir(exist_ok=True)
        logger.debug(f"Папка {folder} проверена/создана")

    logger.debug("Инициализация Eel")
    eel.init('web', allowed_extensions=['.js', '.html', '.css'])
    logger.info("Eel инициализирован, запуск веб-интерфейса")
    eel.start('index.html', size=(1600, 1000), port=0)
