import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ..config_manager.config_loader import ConfigManager
from ..table_builder.table_normalizer import TableBuilder
from ..utils.text_normalizer import TextNormalizer
from ..utils.inn_validator import INNValidator
from .contextual_validator import ContextualFieldValidator
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, NER disabled.")

logger = logging.getLogger(__name__)

@dataclass
class ExtractedField:
    value: Any
    confidence: float
    source: str

class FieldExtractor:
    def __init__(self, config_manager: ConfigManager):
        self.patterns = config_manager.get_patterns()
        self.nlp = None
        self.text_normalizer = TextNormalizer()
        self.inn_validator = INNValidator()
        self.contextual_validator = ContextualFieldValidator()
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("ru_core_news_sm")
                logger.info("spaCy NER loaded.")
            except Exception as e:
                logger.warning(f"spaCy loading failed: {e}")

    def extract_fields(self, text: str, doc_type: str) -> Dict[str, ExtractedField]:
        logger.debug(f"Извлечение полей для типа документа: {doc_type}")
        fields = {}
        
        # First, try NER for financial entities
        ner_fields = self._extract_with_ner(text, doc_type)
        logger.debug(f"NER извлек поля: {list(ner_fields.keys())}")
        
        # Собираем все кандидаты для каждого поля (для разрешения конфликтов)
        field_candidates: Dict[str, List[ExtractedField]] = {}
        
        for field_name, pat in self.patterns.items():
            if field_name in ['saldo_start', 'turnover_debit', 'turnover_credit', 'saldo_end'] and doc_type != 'reconciliation_act':
                continue
            
            candidates = []
            
            # Use NER if available
            if field_name in ner_fields and ner_fields[field_name].confidence > 0.5:
                candidates.append(ner_fields[field_name])
                logger.debug(f"Поле {field_name} найдено через NER: {ner_fields[field_name].value}")
            
            # Также пробуем regex
            regex_field = self._extract_single_field(text, pat, field_name, doc_type)
            if regex_field.value is not None:
                candidates.append(regex_field)
                logger.debug(f"Поле {field_name} найдено через regex: {regex_field.value}")
            
            # Сохраняем кандидатов
            if candidates:
                field_candidates[field_name] = candidates
        
        # Разрешаем конфликты и выбираем лучшие совпадения
        fields = self._resolve_field_conflicts(field_candidates)
        
        logger.debug(f"Извлечение полей завершено: {len(fields)} полей")
        return fields
    
    def _resolve_field_conflicts(self, field_candidates: Dict[str, List[ExtractedField]]) -> Dict[str, ExtractedField]:
        """
        Разрешение конфликтов при множественных совпадениях.
        
        Args:
            field_candidates: Словарь с кандидатами для каждого поля
        
        Returns:
            Словарь с лучшими совпадениями
        """
        resolved = {}
        
        for field_name, candidates in field_candidates.items():
            if not candidates:
                continue
            
            # Сортируем по confidence
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            
            # Берем лучшее совпадение
            best = candidates[0]
            
            # Если есть другие с близким confidence (>90% от лучшего), проверяем контекст
            if len(candidates) > 1 and candidates[1].confidence > best.confidence * 0.9:
                # Выбираем то, которое имеет более длинный source (больше контекста)
                for candidate in candidates[1:]:
                    if candidate.confidence > best.confidence * 0.9:
                        if len(candidate.source) > len(best.source):
                            best = candidate
            
            resolved[field_name] = best
        
        return resolved

    def _extract_single_field(self, text: str, pat: Dict[str, Any], field_name: str = "", doc_type: str = "") -> ExtractedField:
        # Нормализуем текст для лучшего поиска
        normalized_text = TextNormalizer.normalize_for_ocr(text)
        
        # Поддерживаем как старый формат (один паттерн), так и новый (список паттернов с приоритетами)
        if 'patterns' in pat:
            patterns = pat['patterns']
            # Сортируем по приоритету (высший приоритет первым)
            patterns = sorted(patterns, key=lambda x: x.get('priority', 5), reverse=True)
        else:
            # Старый формат - один паттерн
            patterns = [{'pattern': pat['pattern'], 'priority': 5, 'context_keywords': []}]
        
        validate = pat['validate']
        best_match = None
        best_value = None
        best_confidence = 0.0
        best_source = ""
        
        for pattern_config in patterns:
            pattern = pattern_config['pattern']
            
            # Compile the pattern if it's a raw string
            if isinstance(pattern, str) and pattern.startswith('r\'') and pattern.endswith('\''):
                pattern = pattern[2:-1]  # Remove r' and '
            
            # Ищем в нормализованном тексте
            matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE | re.MULTILINE))
            
            for match in matches:
                raw_value = match.group(1).strip()
                
                # Проверяем контекстные ключевые слова
                context_keywords = pattern_config.get('context_keywords', [])
                if context_keywords:
                    # Ищем ключевые слова в радиусе 200 символов
                    start = max(0, match.start() - 200)
                    end = min(len(normalized_text), match.end() + 200)
                    context_text = normalized_text[start:end]
                    
                    # Если нет ключевых слов в контексте, пропускаем
                    if not any(re.search(kw, context_text, re.IGNORECASE) for kw in context_keywords):
                        continue
                
                # Clean the raw value (для чисел убираем пробелы, для текста сохраняем)
                if validate in ['digits', 'digits_9', 'digits_10_12', 'digits_20', 'float']:
                    raw_value = raw_value.replace(" ", "").replace("\u00A0", "").replace("\t", "").replace("\n", "")
                else:
                    raw_value = raw_value.replace("\u00A0", " ").replace("\t", " ").replace("\n", " ").strip()
                
                value = self._normalize_value(raw_value, validate)
                
                if value is not None:
                    # Получаем позицию в оригинальном тексте
                    match_position = match.start()
                    
                    # Получаем контекст
                    context = self.contextual_validator.validate_spatial_context(
                        str(value), field_name, text, match_position, doc_type
                    )
                    
                    # Базовая confidence
                    base_confidence = self._calculate_extraction_confidence(match.group(0), validate)
                    
                    # Учитываем приоритет паттерна
                    priority_boost = pattern_config.get('priority', 5) / 100
                    
                    # Учитываем контекст
                    contextual_confidence = self.contextual_validator.calculate_contextual_confidence(
                        base_confidence, context
                    )
                    
                    confidence = min(1.0, contextual_confidence + priority_boost)
                    
                    # Специальная валидация для ИНН
                    if field_name in ['inn', 'supplier_inn', 'buyer_inn', 'payer_inn', 'recipient_inn', 
                                     'seller_inn', 'executor_inn', 'customer_inn', 'party1_inn', 'party2_inn']:
                        validated_inn, adjusted_confidence = self.inn_validator.validate_and_adjust_confidence(
                            str(value), confidence
                        )
                        if validated_inn:
                            value = validated_inn
                            confidence = adjusted_confidence
                        else:
                            # ИНН не прошел валидацию, пропускаем
                            continue
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_value = value
                        best_match = match.group(0)
                        best_source = match.group(0)
        
        if best_value is not None:
            return ExtractedField(best_value, best_confidence, best_source)
        else:
            return ExtractedField(None, 0.0, "")
    
    def _calculate_extraction_confidence(self, matched_text: str, validate_type: str) -> float:
        """Calculate confidence based on the quality of the match"""
        if validate_type == 'digits_10_12':
            # For INN, longer match is better
            return min(1.0, len(matched_text) / 20)
        elif validate_type == 'digits_9':
            # For KPP, exact match is important
            digits = re.sub(r'[\\D]', '', matched_text)
            if len(digits) == 9:
                return 1.0
            else:
                return 0.5
        elif validate_type == 'float':
            # For amounts, more digits after decimal suggest precision
            return min(1.0, len(matched_text) / 20)
        elif validate_type == 'date':
            # For dates, standard formats get higher confidence
            return 0.9
        else:
            return 0.7

    def _normalize_value(self, raw: str, validate: str) -> Any:
        try:
            if validate == 'digits_10_12':
                if re.match(r'^\d{10,12}$', raw):
                    return raw
            elif validate == 'digits_9':
                if re.match(r'^\d{9}$', raw):
                    return raw
            elif validate == 'digits':
                # Любое число
                digits_only = re.sub(r'\D', '', raw)
                if digits_only:
                    return digits_only
            elif validate == 'digits_20':
                if re.match(r'^\d{20}$', raw):
                    return raw
            elif validate == 'text':
                # Текст - очищаем и возвращаем
                cleaned = raw.strip()
                # Убираем лишние пробелы
                cleaned = re.sub(r'\s+', ' ', cleaned)
                return cleaned if cleaned else None
            elif validate == 'float':
                # Handle various number formats (with commas as decimal separators, etc.)
                cleaned = raw.replace(',', '.').replace(' ', '').replace('\u00A0', '')
                cleaned = re.sub(r'[^\d.-]', '', cleaned)
                # Проверяем, что это не пустая строка или только точка/минус
                if cleaned and cleaned not in ['.', '-', '-.', '.-'] and cleaned.replace('.', '').replace('-', ''):
                    try:
                        return float(cleaned)
                    except ValueError:
                        return None
            elif validate == 'date':
                # Handle multiple date formats
                date_patterns = [
                    (r'(\d{2})\.(\d{2})\.(\d{4})', '%d.%m.%Y'),  # DD.MM.YYYY
                    (r'(\d{2})\s+(\d{2})\s+(\d{4})', '%d %m %Y'),  # DD MM YYYY
                    (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),  # YYYY-MM-DD
                    (r'(\d{2})-(\d{2})-(\d{4})', '%d-%m-%Y'),  # DD-MM-YYYY
                ]
                
                for pattern, fmt in date_patterns:
                    match = re.search(pattern, raw)
                    if match:
                        try:
                            dt = datetime.strptime(match.group(0), fmt)
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                
                # Try to parse Russian date format with month names
                try:
                    # Replace Russian month names with English equivalents for parsing
                    month_names = {
                        'января': 'January', 'февраля': 'February', 'марта': 'March',
                        'апреля': 'April', 'мая': 'May', 'июня': 'June',
                        'июля': 'July', 'августа': 'August', 'сентября': 'September',
                        'октября': 'October', 'ноября': 'November', 'декабря': 'December'
                    }
                    
                    date_str = raw.lower()
                    for ru_month, en_month in month_names.items():
                        date_str = date_str.replace(ru_month, en_month)
                    
                    # Try parsing date with month name
                    dt = datetime.strptime(date_str, '%d %B %Y')
                    return dt.strftime('%Y-%m-%d')
                except ValueError as ve:
                    logger.debug(f"Failed to parse Russian date format '{raw}': {ve}")
                    return None
                    
        except Exception as e:
            logger.warning(f"Value normalization error for '{raw}' with validation '{validate}': {e}")
            return None
        return None

    def _extract_with_ner(self, text: str, doc_type: str = "") -> Dict[str, ExtractedField]:
        """Расширенное извлечение с NER для всех типов сущностей."""
        fields = {}
        if not self.nlp:
            logger.debug("NER недоступен")
            return fields
        
        logger.debug("Запуск NER для извлечения сущностей")
        doc = self.nlp(text)
        
        # Организации (ORG)
        orgs = [ent for ent in doc.ents if ent.label_ == "ORG"]
        if orgs:
            # Первая организация - поставщик/продавец
            if len(orgs) >= 1:
                org_text = orgs[0].text.strip()
                if len(org_text) > 3:  # Минимальная длина
                    fields['supplier'] = ExtractedField(org_text, 0.85, orgs[0].text)
                    logger.debug(f"Извлечена организация (supplier): {org_text}")
            
            # Вторая организация - покупатель/заказчик
            if len(orgs) >= 2:
                org_text = orgs[1].text.strip()
                if len(org_text) > 3:
                    fields['buyer'] = ExtractedField(org_text, 0.85, orgs[1].text)
                    logger.debug(f"Извлечена организация (buyer): {org_text}")
        
        # Персоны (PERSON)
        persons = [ent for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            person_text = persons[0].text.strip()
            if len(person_text) > 3:
                # Определяем тип поля в зависимости от типа документа
                if doc_type in ['dismissal_act', 'employment_contract', 'tax_certificate']:
                    fields['employee_name'] = ExtractedField(person_text, 0.9, persons[0].text)
                    logger.debug(f"Извлечена персона (employee_name): {person_text}")
                elif doc_type in ['medical_report']:
                    fields['patient_name'] = ExtractedField(person_text, 0.9, persons[0].text)
                    logger.debug(f"Извлечена персона (patient_name): {person_text}")
        
        # Деньги (MONEY)
        money_entities = [ent for ent in doc.ents if ent.label_ == "MONEY"]
        if money_entities:
            # Берем самое большое значение как основную сумму
            amounts = []
            for ent in money_entities:
                value = self._normalize_value(ent.text, 'float')
                if value is not None:
                    amounts.append((value, ent.text))
            
            if amounts:
                # Сортируем по значению и берем максимальное
                amounts.sort(key=lambda x: x[0], reverse=True)
                max_amount, source = amounts[0]
                fields["amount"] = ExtractedField(max_amount, 0.9, source)
                logger.debug(f"Извлечено MONEY: {max_amount}")
        
        # Даты (DATE)
        date_entities = [ent for ent in doc.ents if ent.label_ == "DATE"]
        if date_entities:
            # Берем первую найденную дату
            for ent in date_entities:
                value = self._normalize_value(ent.text, 'date')
                if value is not None:
                    fields["date"] = ExtractedField(value, 0.9, ent.text)
                    logger.debug(f"Извлечено DATE: {value}")
                    break
        
        # Локации (LOC) - для адресов
        locations = [ent for ent in doc.ents if ent.label_ == "LOC"]
        if locations:
            loc_text = locations[0].text.strip()
            if len(loc_text) > 5:  # Адреса обычно длиннее
                fields['address'] = ExtractedField(loc_text, 0.7, locations[0].text)
                logger.debug(f"Извлечена локация (address): {loc_text}")
        
        logger.debug(f"NER завершено: извлечено {len(fields)} полей")
        return fields

class SemanticParser:
    def __init__(self, config_manager: ConfigManager, table_builder: TableBuilder):
        self.field_extractor = FieldExtractor(config_manager)
        self.table_builder = table_builder
        self.contextual_validator = ContextualFieldValidator()
        self.document_parsers = {
            'invoice': self._parse_invoice,
            'payment_order': self._parse_payment_order,
            'act': self._parse_act,
            'invoice_factura': self._parse_invoice_factura,
            'dismissal_act': self._parse_dismissal_act,
            'reconciliation_act': self._parse_reconciliation_act,
            'medical_report': self._parse_medical_report,
            'upd': self._parse_upd,
            'torg12': self._parse_torg12,
            'contract': self._parse_contract,
            'receipt': self._parse_receipt,
            'advance_report': self._parse_advance_report,
            'transport_note': self._parse_transport_note,
            'corrective_invoice': self._parse_corrective_invoice,
            'corrective_upd': self._parse_corrective_upd,
            'power_of_attorney': self._parse_power_of_attorney,
            'cash_receipt_order': self._parse_cash_receipt_order,
            'cash_expense_order': self._parse_cash_expense_order,
            'payroll_statement': self._parse_payroll_statement,
            'employment_contract': self._parse_employment_contract,
            'tax_certificate': self._parse_tax_certificate,
            'ticket': self._parse_ticket
        }

    def parse_document(self, text: str, structure: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        parsed = self.document_parsers.get(doc_type, self._parse_generic)(text, structure)
        
        # Извлекаем поля из таблиц и добавляем к существующим
        # Таблицы могут быть в разных форматах: DataFrame или уже to_dict('records')
        if 'tables' in parsed:
            # Конвертируем таблицы обратно в DataFrame для извлечения полей
            try:
                import pandas as pd
                tables_for_extraction = []
                for table in parsed['tables']:
                    if isinstance(table, list) and table and isinstance(table[0], dict):
                        # Это уже to_dict('records')
                        df = pd.DataFrame(table)
                        tables_for_extraction.append(df)
                    elif isinstance(table, pd.DataFrame):
                        tables_for_extraction.append(table)
                
                if tables_for_extraction:
                    table_fields = self._extract_from_tables(tables_for_extraction, doc_type)
                    if 'fields' in parsed:
                        # Конвертируем ExtractedField в dict формат
                        for field_name, field_value in table_fields.items():
                            if field_name not in parsed['fields'] or parsed['fields'][field_name].get('value') is None:
                                if hasattr(field_value, '__dict__'):
                                    parsed['fields'][field_name] = field_value.__dict__
                                else:
                                    parsed['fields'][field_name] = {'value': field_value, 'confidence': 0.8, 'source': 'table'}
            except Exception as e:
                logger.debug(f"Ошибка при извлечении полей из таблиц: {e}")
        
        # Apply consistency validation
        if 'fields' in parsed:
            # Конвертируем обратно в dict для валидации
            fields_dict = {}
            for k, v in parsed['fields'].items():
                if isinstance(v, dict):
                    fields_dict[k] = v
                elif hasattr(v, '__dict__'):
                    fields_dict[k] = v.__dict__
                else:
                    fields_dict[k] = {'value': v, 'confidence': 1.0, 'source': ''}
            
            # Применяем контекстную валидацию
            validated_fields = self.contextual_validator.validate_field_relationships(fields_dict, doc_type)
            parsed['fields'] = validated_fields
            
            # Также применяем старую валидацию для обратной совместимости
            parsed['fields'] = self._validate_consistency(parsed['fields'], doc_type)
        
        return parsed
    
    def _extract_from_tables(self, tables: List[Any], doc_type: str) -> Dict[str, ExtractedField]:
        """
        Извлечение полей из таблиц.
        
        Args:
            tables: Список таблиц (может быть List[pd.DataFrame] или List[List[Dict]])
            doc_type: Тип документа
        
        Returns:
            Словарь извлеченных полей
        """
        fields = {}
        
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas не доступен для извлечения из таблиц")
            return fields
        
        for table in tables:
            df = None
            
            # Обрабатываем разные форматы таблиц
            if isinstance(table, pd.DataFrame):
                df = table
            elif isinstance(table, list):
                # Может быть список словарей (из to_dict('records'))
                if table and isinstance(table[0], dict):
                    df = pd.DataFrame(table)
                # Или список списков (сырые данные)
                elif table and isinstance(table[0], (list, tuple)):
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0] if table else None)
                    except Exception:
                        continue
            elif isinstance(table, dict):
                # Один словарь - конвертируем в DataFrame
                try:
                    df = pd.DataFrame([table])
                except Exception:
                    continue
            
            if df is None or df.empty:
                continue
            
            # Ищем ИНН в таблицах
            for col in df.columns:
                col_str = str(col).lower()
                if 'инн' in col_str:
                    for val in df[col].dropna():
                        val_str = str(val).strip()
                        # Убираем нецифровые символы для проверки
                        digits_only = re.sub(r'\D', '', val_str)
                        # Проверяем формат ИНН
                        if re.match(r'^\d{10,12}$', digits_only):
                            # Валидируем ИНН
                            is_valid, _ = self.field_extractor.inn_validator.validate_inn(digits_only)
                            if is_valid:
                                if 'supplier_inn' not in fields:
                                    fields['supplier_inn'] = ExtractedField(digits_only, 0.8, f"Таблица: {col}")
                                elif 'buyer_inn' not in fields:
                                    fields['buyer_inn'] = ExtractedField(digits_only, 0.8, f"Таблица: {col}")
                            break
            
            # Ищем КПП
            for col in df.columns:
                col_str = str(col).lower()
                if 'кпп' in col_str:
                    for val in df[col].dropna():
                        val_str = str(val).strip()
                        if re.match(r'^\d{9}$', val_str):
                            if 'kpp' not in fields:
                                fields['kpp'] = ExtractedField(val_str, 0.8, f"Таблица: {col}")
                            break
            
            # Ищем суммы
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            for col in numeric_cols:
                col_str = str(col).lower()
                if any(keyword in col_str for keyword in ['сумма', 'итого', 'всего', 'к оплате', 'стоимость']):
                    # Берем последнее значение (обычно это итог)
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(values) > 0:
                        total = float(values.iloc[-1])
                        if total > 0:
                            if 'amount' not in fields or fields['amount'].confidence < 0.8:
                                fields['amount'] = ExtractedField(total, 0.9, f"Таблица: {col}")
                            elif 'total_with_vat' not in fields:
                                fields['total_with_vat'] = ExtractedField(total, 0.9, f"Таблица: {col}")
            
            # Ищем БИК
            for col in df.columns:
                col_str = str(col).lower()
                if 'бик' in col_str:
                    for val in df[col].dropna():
                        val_str = str(val).strip()
                        if re.match(r'^\d{9}$', val_str):
                            if 'bic' not in fields:
                                fields['bic'] = ExtractedField(val_str, 0.8, f"Таблица: {col}")
                            break
        
        return fields

    def _parse_generic(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        fields = self.field_extractor.extract_fields(text, 'generic')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        
        # Извлекаем дополнительные поля из текста
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _extract_additional_fields(self, text: str) -> Dict[str, ExtractedField]:
        """Извлекает дополнительные поля из текста, не определенные в паттернах."""
        additional = {}
        
        # Извлекаем названия организаций
        org_pattern = r'(?:ООО|ОАО|ЗАО|ИП|ПАО|АО|ИНН)\s*[«"]?([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+?)(?:[«"]|ИНН|,|$)'
        org_matches = re.finditer(org_pattern, text, re.IGNORECASE)
        for match in list(org_matches)[:2]:  # Берем первые 2
            org_name = match.group(1).strip()
            if len(org_name) > 3:
                additional[f'organization_{len(additional)+1}'] = ExtractedField(org_name, 0.7, match.group(0))
        
        return additional

    def _parse_invoice(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Расширенный парсинг счетов на оплату с извлечением всех полей."""
        fields = self.field_extractor.extract_fields(text, 'invoice')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        
        # Извлекаем дополнительные поля из текста
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        
        # Извлекаем дополнительную информацию из таблиц
        table_fields = {}
        items_list = []
        tables_for_extraction = []
        
        for df in tables:
            if not df.empty:
                # Извлекаем данные товаров/услуг
                items = df.to_dict('records')
                items_list.extend(items)
                # Сохраняем DataFrame для извлечения полей
                tables_for_extraction.append(df)
        
        if items_list:
            table_fields['items'] = items_list
        
        # Извлекаем поля из текста более детально
        invoice_specific = self._extract_invoice_specific_fields(text)
        fields.update(invoice_specific)
        
        # Если поставщик/покупатель не найдены, используем organization_1 и organization_2
        if 'supplier' not in fields or fields['supplier'].value is None:
            if 'organization_1' in fields and fields['organization_1'].value:
                org1_value = fields['organization_1'].value
                if isinstance(org1_value, str) and len(org1_value.strip()) > 2:
                    # Очищаем от лишних символов
                    org1_clean = org1_value.strip().rstrip('»').rstrip(',').strip()
                    if org1_clean:
                        fields['supplier'] = ExtractedField(org1_clean, 0.7, fields['organization_1'].source)
        
        if 'buyer' not in fields or fields['buyer'].value is None:
            if 'organization_2' in fields and fields['organization_2'].value:
                org2_value = fields['organization_2'].value
                if isinstance(org2_value, str) and len(org2_value.strip()) > 2:
                    # Очищаем от лишних символов
                    org2_clean = org2_value.strip().rstrip('»').rstrip(',').strip()
                    if org2_clean:
                        fields['buyer'] = ExtractedField(org2_clean, 0.7, fields['organization_2'].source)
        
        # Если ИНН не найден напрямую, пытаемся найти рядом с названием организации
        if 'supplier_inn' not in fields or fields['supplier_inn'].value is None:
            if 'supplier' in fields and fields['supplier'].value:
                # Ищем ИНН после названия поставщика
                supplier_name = fields['supplier'].value
                supplier_inn_match = re.search(
                    re.escape(supplier_name[:20]) + r'[^\d]*ИНН\s*[:\-]?\s*(\d{10,12})',
                    text,
                    re.IGNORECASE
                )
                if supplier_inn_match:
                    fields['supplier_inn'] = ExtractedField(supplier_inn_match.group(1), 0.8, supplier_inn_match.group(0))
        
        if 'buyer_inn' not in fields or fields['buyer_inn'].value is None:
            if 'buyer' in fields and fields['buyer'].value:
                # Ищем ИНН после названия покупателя
                buyer_name = fields['buyer'].value
                buyer_inn_match = re.search(
                    re.escape(buyer_name[:20]) + r'[^\d]*ИНН\s*[:\-]?\s*(\d{10,12})',
                    text,
                    re.IGNORECASE
                )
                if buyer_inn_match:
                    fields['buyer_inn'] = ExtractedField(buyer_inn_match.group(1), 0.8, buyer_inn_match.group(0))
        
        # Улучшаем извлечение даты
        if 'date' not in fields or fields['date'].value is None:
            # Пробуем разные форматы даты
            date_patterns = [
                r'(?:дата|от|дата\s+счета)[\s:‑\-\/]*(\d{1,2}[\.\s\/]\d{1,2}[\.\s\/]\d{4})',
                r'(\d{1,2}[\.\s\/]\d{1,2}[\.\s\/]\d{4})',
                r'(\d{1,2}\s+[а-яё]+\s+20\d{2})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, text, re.IGNORECASE)
                if date_match:
                    date_str = date_match.group(1)
                    # Проверяем, что это не номер счета или ИНН
                    if not re.match(r'^\d{10,12}$', date_str.replace('.', '').replace(' ', '').replace('/', '')):
                        fields['date'] = ExtractedField(date_str, 0.7, date_match.group(0))
                        break
        
        # Извлекаем поля из таблиц
        table_extracted_fields = self._extract_from_tables(tables_for_extraction, 'invoice')
        for field_name, field_value in table_extracted_fields.items():
            if field_name not in fields or fields[field_name].value is None:
                fields[field_name] = field_value
        
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables],
            'table_fields': table_fields
        }
    
    def _extract_invoice_specific_fields(self, text: str) -> Dict[str, ExtractedField]:
        """Извлекает специфичные поля для счетов на оплату."""
        fields = {}
        
        # Номер счета (улучшенный паттерн для случаев типа "__10" или "N 10")
        doc_num_patterns = [
            r'счет\s*(?:на\s+оплату)?\s*[№N]\s*[:\-]?\s*_+(\d+)',  # СЧЕТ НА ОПЛАТУ N __10
            r'счёт\s*(?:на\s+оплату)?\s*[№N]\s*[:\-]?\s*_+(\d+)',
            r'счет\s*(?:на\s+оплату)?\s*[№N]\s*[:\-]?\s*[_\s]+(\d+)',  # СЧЕТ НА ОПЛАТУ N __10 (с пробелами)
            r'счёт\s*(?:на\s+оплату)?\s*[№N]\s*[:\-]?\s*[_\s]+(\d+)',
            r'счет\s*(?:на\s+оплату)?\s*[№N]?\s*[:\-]?\s*№?\s*(\d+)',  # Обычный формат
            r'счёт\s*(?:на\s+оплату)?\s*[№N]?\s*[:\-]?\s*№?\s*(\d+)',
            r'счет\s*[№N]\s*[:\-]?\s*(\d+)',
            r'счет\s*на\s+оплату\s*[№N]?\s*[:\-]?\s*(\d+)',
            r'счет\s*№\s*(\d+)',
            r'счёт\s*№\s*(\d+)'
        ]
        for pattern in doc_num_patterns:
            doc_num_match = re.search(pattern, text, re.IGNORECASE)
            if doc_num_match:
                doc_num = doc_num_match.group(1)
                # Игнорируем "0" и слишком короткие номера
                if doc_num and doc_num != '0' and len(doc_num) >= 1:
                    fields['document_number'] = ExtractedField(doc_num, 0.9, doc_num_match.group(0))
                    break
        
        # Поставщик (улучшенный паттерн)
        supplier_patterns = [
            r'поставщик[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+?)(?:\n|ИНН|КПП|адрес|тел|$|БИК|р\/с)',
            r'поставщик[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+?)(?=\s+ИНН|\s+КПП|\s+БИК|\s+р\/с|$)',
            r'поставщик[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+)'
        ]
        for pattern in supplier_patterns:
            supplier_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if supplier_match:
                supplier_name = supplier_match.group(1).strip().rstrip(',')
                if len(supplier_name) > 2:  # Минимальная длина
                    fields['supplier'] = ExtractedField(supplier_name, 0.8, supplier_match.group(0))
                    break
        
        # Покупатель/Заказчик (улучшенный паттерн)
        buyer_patterns = [
            r'(?:покупатель|заказчик)[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+?)(?:\n|ИНН|КПП|адрес|тел|$|БИК|р\/с)',
            r'(?:покупатель|заказчик)[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+?)(?=\s+ИНН|\s+КПП|\s+БИК|\s+р\/с|$)',
            r'(?:покупатель|заказчик)[:\s]+([А-ЯЁ][А-ЯЁа-яё\s«»\-,\.]+)'
        ]
        for pattern in buyer_patterns:
            buyer_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if buyer_match:
                buyer_name = buyer_match.group(1).strip().rstrip(',')
                if len(buyer_name) > 2:  # Минимальная длина
                    fields['buyer'] = ExtractedField(buyer_name, 0.8, buyer_match.group(0))
                    break
        
        # ИНН поставщика (улучшенный паттерн)
        supplier_inn_patterns = [
            r'поставщик[:\s]+[А-ЯЁ\s«»,\.\-]+?ИНН\s*(\d{10,12})',
            r'ИНН\s+поставщика[:\s]*(\d{10,12})',
            r'поставщик[:\s]+[А-ЯЁ\s«»,\.\-]+?\s+ИНН\s*[:\-]?\s*(\d{10,12})'
        ]
        for pattern in supplier_inn_patterns:
            supplier_inn_match = re.search(pattern, text, re.IGNORECASE)
            if supplier_inn_match:
                fields['supplier_inn'] = ExtractedField(supplier_inn_match.group(1), 0.9, supplier_inn_match.group(0))
                break
        
        # ИНН покупателя (улучшенный паттерн)
        buyer_inn_patterns = [
            r'(?:покупатель|заказчик)[:\s]+[А-ЯЁ\s«»,\.\-]+?ИНН\s*(\d{10,12})',
            r'ИНН\s+(?:покупателя|заказчика)[:\s]*(\d{10,12})',
            r'(?:покупатель|заказчик)[:\s]+[А-ЯЁ\s«»,\.\-]+?\s+ИНН\s*[:\-]?\s*(\d{10,12})'
        ]
        for pattern in buyer_inn_patterns:
            buyer_inn_match = re.search(pattern, text, re.IGNORECASE)
            if buyer_inn_match:
                fields['buyer_inn'] = ExtractedField(buyer_inn_match.group(1), 0.9, buyer_inn_match.group(0))
                break
        
        # БИК
        bic_match = re.search(r'БИК\s*[:\-]?\s*(\d{9})', text, re.IGNORECASE)
        if bic_match:
            fields['bic'] = ExtractedField(bic_match.group(1), 0.9, bic_match.group(0))
        
        # НДС
        vat_match = re.search(r'(?:ндс|н\s*д\s*с)[\s:‑\-\/]*(?:в\s+том\s+числе\s+)?([\d\s,\.]+)', text, re.IGNORECASE)
        if vat_match:
            vat_value = self.field_extractor._normalize_value(vat_match.group(1).replace(' ', ''), 'float')
            if vat_value:
                fields['vat'] = ExtractedField(vat_value, 0.8, vat_match.group(0))
        
        return fields
    
    def _parse_medical_report(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг медицинских отчетов."""
        fields = self.field_extractor.extract_fields(text, 'medical_report')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        
        return {
            'fields': {k: v.__dict__ for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_reconciliation_act(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        fields = self.field_extractor.extract_fields(text, 'reconciliation_act')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        operations = {}
        for df in tables:
            ops = self.table_builder.extract_operations(df)
            if ops:
                operations.update(ops)
        return {
            'fields': {k: v.__dict__ for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables],
            'operations': operations
        }
    
    def _parse_payment_order(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг платежных поручений."""
        fields = self.field_extractor.extract_fields(text, 'payment_order')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_act(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг актов выполненных работ/услуг."""
        fields = self.field_extractor.extract_fields(text, 'act')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_invoice_factura(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг счетов-фактур."""
        fields = self.field_extractor.extract_fields(text, 'invoice_factura')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_dismissal_act(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг приказов об увольнении."""
        fields = self.field_extractor.extract_fields(text, 'dismissal_act')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_upd(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг УПД (универсальный передаточный документ)."""
        fields = self.field_extractor.extract_fields(text, 'upd')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_torg12(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг товарных накладных ТОРГ-12."""
        fields = self.field_extractor.extract_fields(text, 'torg12')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_contract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг договоров."""
        fields = self.field_extractor.extract_fields(text, 'contract')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        additional_fields = self._extract_additional_fields(text)
        fields.update(additional_fields)
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_receipt(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг кассовых чеков."""
        fields = self.field_extractor.extract_fields(text, 'receipt')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_advance_report(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг авансовых отчетов."""
        fields = self.field_extractor.extract_fields(text, 'advance_report')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_transport_note(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг транспортных накладных."""
        fields = self.field_extractor.extract_fields(text, 'transport_note')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_corrective_invoice(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг корректировочных счетов-фактур."""
        fields = self.field_extractor.extract_fields(text, 'corrective_invoice')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_corrective_upd(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг корректировочных УПД."""
        fields = self.field_extractor.extract_fields(text, 'corrective_upd')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_power_of_attorney(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг доверенностей."""
        fields = self.field_extractor.extract_fields(text, 'power_of_attorney')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_cash_receipt_order(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг приходных кассовых ордеров."""
        fields = self.field_extractor.extract_fields(text, 'cash_receipt_order')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_cash_expense_order(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг расходных кассовых ордеров."""
        fields = self.field_extractor.extract_fields(text, 'cash_expense_order')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_payroll_statement(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг зарплатных ведомостей."""
        fields = self.field_extractor.extract_fields(text, 'payroll_statement')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_employment_contract(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг трудовых договоров."""
        fields = self.field_extractor.extract_fields(text, 'employment_contract')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_tax_certificate(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг справок 2-НДФЛ и расчетных листков."""
        fields = self.field_extractor.extract_fields(text, 'tax_certificate')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }
    
    def _parse_ticket(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Парсинг электронных билетов."""
        fields = self.field_extractor.extract_fields(text, 'ticket')
        tables = self.table_builder.normalize_tables([page['tables'] for page in structure['pages']])
        return {
            'fields': {k: v.__dict__ if hasattr(v, '__dict__') else {'value': v, 'confidence': 1.0, 'source': ''} for k, v in fields.items()},
            'tables': [df.to_dict('records') for df in tables]
        }

    def _validate_consistency(self, fields: Dict[str, Dict[str, Any]], doc_type: str) -> Dict[str, Dict[str, Any]]:
        if doc_type == 'reconciliation_act':
            fields = self._validate_and_correct_saldo(fields)
        # Add more validations as needed
        return fields

    def _validate_and_correct_saldo(self, fields: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        saldo_start = fields.get('saldo_start', {}).get('value')
        debit = fields.get('turnover_debit', {}).get('value')
        credit = fields.get('turnover_credit', {}).get('value')
        saldo_end = fields.get('saldo_end', {}).get('value')
        if all(v is not None for v in [saldo_start, debit, credit, saldo_end]):
            expected = round(saldo_start + debit - credit, 2)
            if abs(fields['saldo_end']['value'] - expected) > 0.01:
                fields['saldo_end']['warning'] = f"Несоответствие: {fields['saldo_end']['value']} ≠ {expected}"
                fields['saldo_end']['auto_corrected'] = expected
        return fields
