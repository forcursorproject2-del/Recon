"""
Модуль для работы с историей обработки документов.
Хранит информацию о всех обработанных документах в SQLite БД.
"""
import sqlite3
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


class ProcessingHistory:
    """Класс для работы с историей обработки документов."""
    
    def __init__(self, db_path: str = "processing_history.db"):
        """
        Инициализация БД истории обработки.
        
        Args:
            db_path: Путь к файлу БД SQLite
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self):
        """Создание таблиц БД, если их нет."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                document_type TEXT,
                status TEXT NOT NULL,
                contractor_name TEXT,
                contractor_inn TEXT,
                document_date TEXT,
                document_number TEXT,
                total_amount REAL,
                amount_with_vat REAL,
                amount_without_vat REAL,
                processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                output_folder TEXT,
                output_files TEXT,
                file_hash TEXT,
                error_message TEXT,
                confidence REAL,
                UNIQUE(original_filename, file_hash)
            )
        """)
        
        # Индексы для быстрого поиска
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contractor_inn ON documents(contractor_inn)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_type ON documents(document_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON documents(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed_at ON documents(processed_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_date ON documents(document_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_contractor_name ON documents(contractor_name)
        """)
        
        self.conn.commit()
        logger.debug("БД истории обработки инициализирована")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Вычисляет MD5 хэш файла для детекта дубликатов."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.warning(f"Не удалось вычислить хэш файла {file_path}: {e}")
            return ""
    
    def _is_invalid_contractor_name(self, value: str) -> bool:
        """
        Проверяет, является ли значение некорректным названием контрагента.
        
        Args:
            value: Значение для проверки
        
        Returns:
            True если значение некорректно
        """
        if not value or not isinstance(value, str):
            return True
        
        value_lower = value.lower().strip()
        
        # Список некорректных значений
        invalid_values = [
            'наличныйрасчет', 'наличный расчет', 'наличный', 'расчет',
            'безналичный', 'безналичный расчет', 'безналичныйрасчет',
            'оплата', 'способ оплаты', 'способоплаты',
            'н/д', 'н/а', 'не указано', 'не указан', 'не указана',
            '-', '—', '–', 'нет', 'отсутствует'
        ]
        
        # Проверяем точное совпадение
        if value_lower in invalid_values:
            return True
        
        # Проверяем частичное совпадение (если значение слишком короткое или содержит только эти слова)
        if len(value_lower) < 3:
            return True
        
        # Проверяем, не является ли это только способом оплаты
        payment_indicators = ['наличн', 'безналичн', 'карт', 'перевод', 'электронн']
        if any(indicator in value_lower for indicator in payment_indicators) and len(value_lower) < 20:
            return True
        
        return False
    
    def _extract_contractor_info(self, doc_data: Dict[str, Any]) -> tuple:
        """
        Извлекает информацию о контрагенте из данных документа.
        
        Returns:
            (contractor_name, contractor_inn)
        """
        fields = doc_data.get('fields', {})
        doc_type = doc_data.get('doc_type', 'unknown')
        
        contractor_name = None
        contractor_inn = None
        
        # Для медицинских документов используем laboratory_name
        if doc_type == 'medical_report':
            if 'laboratory_name' in fields:
                field_value = fields['laboratory_name']
                if isinstance(field_value, dict):
                    value = field_value.get('value')
                else:
                    value = field_value
                
                if value and isinstance(value, str) and value.strip():
                    value = value.strip()
                    if not self._is_invalid_contractor_name(value):
                        contractor_name = value
        
        # Для остальных документов используем стандартные поля
        if not contractor_name:
            # Приоритет полей для определения контрагента
            contractor_fields = [
                'supplier', 'buyer', 'seller', 'payer', 'recipient',
                'executor', 'customer', 'counterparty', 'party1', 'party2'
            ]
            
            # Ищем название контрагента
            for field_name in contractor_fields:
                if field_name in fields:
                    field_value = fields[field_name]
                    if isinstance(field_value, dict):
                        value = field_value.get('value')
                    else:
                        value = field_value
                    
                    if value and isinstance(value, str) and value.strip():
                        value = value.strip()
                        # Проверяем, что значение корректно
                        if not self._is_invalid_contractor_name(value):
                            contractor_name = value
                            break
        
        # Ищем ИНН контрагента
        inn_fields = [
            'supplier_inn', 'buyer_inn', 'seller_inn', 'payer_inn', 'recipient_inn',
            'executor_inn', 'customer_inn', 'counterparty_inn', 'party1_inn', 'party2_inn'
        ]
        
        for field_name in inn_fields:
            if field_name in fields:
                field_value = fields[field_name]
                if isinstance(field_value, dict):
                    value = field_value.get('value')
                else:
                    value = field_value
                
                if value and isinstance(value, str) and value.strip():
                    # Проверяем, что это действительно ИНН (10 или 12 цифр)
                    value_clean = value.strip().replace(' ', '').replace('-', '')
                    if value_clean.isdigit() and len(value_clean) in [10, 12]:
                        contractor_inn = value_clean
                        break
        
        return contractor_name, contractor_inn
    
    def _extract_amounts(self, doc_data: Dict[str, Any]) -> tuple:
        """
        Извлекает суммы из данных документа.
        
        Returns:
            (total_amount, amount_with_vat, amount_without_vat)
        """
        fields = doc_data.get('fields', {})
        
        total_amount = None
        amount_with_vat = None
        amount_without_vat = None
        
        # Ищем общую сумму
        if 'amount' in fields:
            field_value = fields['amount']
            if isinstance(field_value, dict):
                value = field_value.get('value')
            else:
                value = field_value
            
            if value:
                try:
                    total_amount = float(str(value).replace(' ', '').replace(',', '.'))
                except (ValueError, AttributeError):
                    pass
        
        # Ищем сумму с НДС
        if 'total_with_vat' in fields:
            field_value = fields['total_with_vat']
            if isinstance(field_value, dict):
                value = field_value.get('value')
            else:
                value = field_value
            
            if value:
                try:
                    amount_with_vat = float(str(value).replace(' ', '').replace(',', '.'))
                except (ValueError, AttributeError):
                    pass
        
        # Ищем сумму без НДС
        if 'total_without_vat' in fields:
            field_value = fields['total_without_vat']
            if isinstance(field_value, dict):
                value = field_value.get('value')
            else:
                value = field_value
            
            if value:
                try:
                    amount_without_vat = float(str(value).replace(' ', '').replace(',', '.'))
                except (ValueError, AttributeError):
                    pass
        
        # Если общая сумма не найдена, используем сумму с НДС или без НДС
        if total_amount is None:
            total_amount = amount_with_vat or amount_without_vat
        
        return total_amount, amount_with_vat, amount_without_vat
    
    def add_document(
        self,
        filename: str,
        original_path: str,
        doc_data: Dict[str, Any],
        output_folder: str,
        output_files: Dict[str, str],
        status: str = "success"
    ) -> int:
        """
        Добавляет запись об обработанном документе в БД.
        
        Args:
            filename: Имя файла
            original_path: Полный путь к исходному файлу
            doc_data: Данные документа (результат парсинга)
            output_folder: Папка с результатами
            output_files: Словарь с путями к выходным файлам {"json": "...", "csv": "...", "xlsx": "..."}
            status: Статус обработки (success, partial, error)
        
        Returns:
            ID добавленной записи
        """
        try:
            # Вычисляем хэш файла
            file_hash = self._calculate_file_hash(original_path)
            
            # Извлекаем информацию о документе
            doc_type = doc_data.get('doc_type', 'unknown')
            confidence = doc_data.get('confidence', 0.0)
            if isinstance(confidence, dict):
                confidence = confidence.get('confidence', 0.0)
            
            # Извлекаем дату и номер документа
            fields = doc_data.get('fields', {})
            document_date = None
            document_number = None
            
            if 'date' in fields:
                date_field = fields['date']
                if isinstance(date_field, dict):
                    document_date = date_field.get('value')
                else:
                    document_date = date_field
            
            if 'document_number' in fields:
                number_field = fields['document_number']
                if isinstance(number_field, dict):
                    document_number = number_field.get('value')
                else:
                    document_number = number_field
            
            # Извлекаем информацию о контрагенте
            contractor_name, contractor_inn = self._extract_contractor_info(doc_data)
            
            # Извлекаем суммы
            total_amount, amount_with_vat, amount_without_vat = self._extract_amounts(doc_data)
            
            # Извлекаем сообщение об ошибке, если есть
            error_message = doc_data.get('error')
            if not error_message and status == "error":
                error_message = "Ошибка обработки"
            
            # Сохраняем пути к выходным файлам как JSON
            output_files_json = json.dumps(output_files, ensure_ascii=False)
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    original_filename, original_path, document_type, status,
                    contractor_name, contractor_inn, document_date, document_number,
                    total_amount, amount_with_vat, amount_without_vat,
                    processed_at, output_folder, output_files, file_hash,
                    error_message, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename, original_path, doc_type, status,
                contractor_name, contractor_inn, document_date, document_number,
                total_amount, amount_with_vat, amount_without_vat,
                datetime.now().isoformat(), output_folder, output_files_json, file_hash,
                error_message, confidence
            ))
            
            doc_id = cursor.lastrowid
            self.conn.commit()
            logger.debug(f"Документ {filename} добавлен в историю (ID: {doc_id})")
            return doc_id
        
        except Exception as e:
            logger.error(f"Ошибка при добавлении документа в историю: {e}", exc_info=True)
            self.conn.rollback()
            return -1
    
    def get_history(
        self,
        page: int = 1,
        per_page: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Получает историю обработки с пагинацией и фильтрами.
        
        Args:
            page: Номер страницы (начиная с 1)
            per_page: Количество записей на странице
            filters: Словарь с фильтрами:
                - search: поиск по имени файла, ИНН, контрагенту, номеру документа
                - document_type: тип документа
                - status: статус обработки
                - contractor_inn: ИНН контрагента
                - date_from: дата документа с
                - date_to: дата документа до
                - processed_from: дата обработки с
                - processed_to: дата обработки до
        
        Returns:
            Словарь с записями и метаданными пагинации
        """
        try:
            filters = filters or {}
            offset = (page - 1) * per_page
            
            # Строим WHERE условие
            where_conditions = []
            params = []
            
            # Поиск по тексту
            if filters.get('search'):
                search = f"%{filters['search']}%"
                where_conditions.append("""
                    (original_filename LIKE ? OR 
                     contractor_name LIKE ? OR 
                     contractor_inn LIKE ? OR 
                     document_number LIKE ?)
                """)
                params.extend([search, search, search, search])
            
            # Фильтр по типу документа
            if filters.get('document_type'):
                where_conditions.append("document_type = ?")
                params.append(filters['document_type'])
            
            # Фильтр по статусу
            if filters.get('status'):
                where_conditions.append("status = ?")
                params.append(filters['status'])
            
            # Фильтр по ИНН контрагента
            if filters.get('contractor_inn'):
                where_conditions.append("contractor_inn = ?")
                params.append(filters['contractor_inn'])
            
            # Фильтр по дате документа
            if filters.get('date_from'):
                where_conditions.append("document_date >= ?")
                params.append(filters['date_from'])
            
            if filters.get('date_to'):
                where_conditions.append("document_date <= ?")
                params.append(filters['date_to'])
            
            # Фильтр по дате обработки
            if filters.get('processed_from'):
                where_conditions.append("processed_at >= ?")
                params.append(filters['processed_from'])
            
            if filters.get('processed_to'):
                where_conditions.append("processed_at <= ?")
                params.append(filters['processed_to'])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Получаем общее количество записей
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM documents WHERE {where_clause}", params)
            total_count = cursor.fetchone()[0]
            
            # Получаем записи с пагинацией
            cursor.execute(f"""
                SELECT * FROM documents 
                WHERE {where_clause}
                ORDER BY processed_at DESC
                LIMIT ? OFFSET ?
            """, params + [per_page, offset])
            
            rows = cursor.fetchall()
            
            # Преобразуем в словари
            documents = []
            for row in rows:
                doc = dict(row)
                # Парсим JSON выходных файлов
                if doc.get('output_files'):
                    try:
                        doc['output_files'] = json.loads(doc['output_files'])
                    except (json.JSONDecodeError, TypeError):
                        doc['output_files'] = {}
                else:
                    doc['output_files'] = {}
                
                documents.append(doc)
            
            return {
                'documents': documents,
                'total': total_count,
                'page': page,
                'per_page': per_page,
                'total_pages': (total_count + per_page - 1) // per_page
            }
        
        except Exception as e:
            logger.error(f"Ошибка при получении истории: {e}", exc_info=True)
            return {
                'documents': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получает статистику по обработанным документам."""
        try:
            cursor = self.conn.cursor()
            
            # Общее количество
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_count = cursor.fetchone()[0]
            
            # За сегодня
            cursor.execute("""
                SELECT COUNT(*) FROM documents 
                WHERE DATE(processed_at) = DATE('now')
            """)
            today_count = cursor.fetchone()[0]
            
            # С ошибками
            cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'error'")
            error_count = cursor.fetchone()[0]
            
            # Процент ошибок
            error_percent = (error_count / total_count * 100) if total_count > 0 else 0
            
            # Самый активный контрагент
            cursor.execute("""
                SELECT contractor_name, COUNT(*) as cnt 
                FROM documents 
                WHERE contractor_name IS NOT NULL 
                GROUP BY contractor_name 
                ORDER BY cnt DESC 
                LIMIT 1
            """)
            top_contractor_row = cursor.fetchone()
            top_contractor = None
            if top_contractor_row:
                top_contractor = {
                    'name': top_contractor_row[0],
                    'count': top_contractor_row[1]
                }
            
            # Статистика по типам документов
            cursor.execute("""
                SELECT document_type, COUNT(*) as cnt 
                FROM documents 
                WHERE document_type IS NOT NULL 
                GROUP BY document_type 
                ORDER BY cnt DESC
            """)
            type_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'total': total_count,
                'today': today_count,
                'errors': error_count,
                'error_percent': round(error_percent, 1),
                'top_contractor': top_contractor,
                'by_type': type_stats
            }
        
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}", exc_info=True)
            return {
                'total': 0,
                'today': 0,
                'errors': 0,
                'error_percent': 0.0,
                'top_contractor': None,
                'by_type': {}
            }
    
    def find_duplicates(self, document_id: int) -> List[Dict[str, Any]]:
        """
        Находит возможные дубликаты документа.
        
        Args:
            document_id: ID документа для проверки
        
        Returns:
            Список возможных дубликатов
        """
        try:
            cursor = self.conn.cursor()
            
            # Получаем информацию о документе
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            doc_row = cursor.fetchone()
            
            if not doc_row:
                return []
            
            doc = dict(doc_row)
            
            # Ищем дубликаты по номеру документа + дате + ИНН
            if doc.get('document_number') and doc.get('document_date') and doc.get('contractor_inn'):
                cursor.execute("""
                    SELECT * FROM documents 
                    WHERE id != ? 
                    AND document_number = ? 
                    AND document_date = ? 
                    AND contractor_inn = ?
                """, (document_id, doc['document_number'], doc['document_date'], doc['contractor_inn']))
                
                duplicates = [dict(row) for row in cursor.fetchall()]
                return duplicates
            
            # Альтернативный поиск по хэшу файла
            if doc.get('file_hash'):
                cursor.execute("""
                    SELECT * FROM documents 
                    WHERE id != ? 
                    AND file_hash = ?
                """, (document_id, doc['file_hash']))
                
                duplicates = [dict(row) for row in cursor.fetchall()]
                return duplicates
            
            return []
        
        except Exception as e:
            logger.error(f"Ошибка при поиске дубликатов: {e}", exc_info=True)
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """Удаляет запись о документе из истории."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            self.conn.commit()
            logger.debug(f"Документ {document_id} удален из истории")
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def cleanup_old_records(self, days: int = 180) -> int:
        """
        Удаляет старые записи из истории.
        
        Args:
            days: Количество дней для хранения истории
        
        Returns:
            Количество удаленных записей
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents WHERE processed_at < ?", (cutoff_date,))
            deleted_count = cursor.rowcount
            self.conn.commit()
            logger.info(f"Удалено {deleted_count} старых записей (старше {days} дней)")
            return deleted_count
        except Exception as e:
            logger.error(f"Ошибка при очистке старых записей: {e}", exc_info=True)
            self.conn.rollback()
            return 0
    
    def get_contractors_list(self) -> List[Dict[str, str]]:
        """Получает список всех контрагентов для фильтра."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT DISTINCT contractor_name, contractor_inn 
                FROM documents 
                WHERE contractor_name IS NOT NULL 
                ORDER BY contractor_name
            """)
            return [{'name': row[0], 'inn': row[1]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Ошибка при получении списка контрагентов: {e}", exc_info=True)
            return []
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Получает документ по ID."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            if row:
                doc = dict(row)
                # Парсим JSON выходных файлов
                if doc.get('output_files'):
                    try:
                        doc['output_files'] = json.loads(doc['output_files'])
                    except (json.JSONDecodeError, TypeError):
                        doc['output_files'] = {}
                else:
                    doc['output_files'] = {}
                return doc
            return None
        except Exception as e:
            logger.error(f"Ошибка при получении документа по ID: {e}", exc_info=True)
            return None
    
    def __del__(self):
        """Закрывает соединение с БД при удалении объекта."""
        try:
            self.conn.close()
        except Exception:
            pass

