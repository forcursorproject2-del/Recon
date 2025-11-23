import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..config_manager.config_loader import ConfigManager
from ..table_builder.table_normalizer import TableBuilder
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
        ner_fields = self._extract_with_ner(text)
        logger.debug(f"NER извлек поля: {list(ner_fields.keys())}")
        for field_name, pat in self.patterns.items():
            if field_name in ['saldo_start', 'turnover_debit', 'turnover_credit', 'saldo_end'] and doc_type != 'reconciliation_act':
                continue
            # Use NER if available, else regex
            if field_name in ner_fields and ner_fields[field_name].confidence > 0.5:
                fields[field_name] = ner_fields[field_name]
                logger.debug(f"Поле {field_name} извлечено с помощью NER: {ner_fields[field_name].value}")
            else:
                fields[field_name] = self._extract_single_field(text, pat)
                logger.debug(f"Поле {field_name} извлечено с помощью regex: {fields[field_name].value}")
        logger.debug(f"Извлечение полей завершено: {len(fields)} полей")
        return fields

    def _extract_single_field(self, text: str, pat: Dict[str, Any]) -> ExtractedField:
        pattern = pat['pattern']
        validate = pat['validate']
        
        # Compile the pattern if it's a raw string
        if isinstance(pattern, str) and pattern.startswith('r\'') and pattern.endswith('\''):
            pattern = pattern[2:-1]  # Remove r' and '
        
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if not matches:
            return ExtractedField(None, 0.0, "")
        
        # Take the best match based on confidence
        best_match = None
        best_value = None
        best_confidence = 0.0
        
        for match in matches:
            raw_value = match.group(1).strip()
            # Clean the raw value
            raw_value = raw_value.replace(" ", "").replace("\u00A0", "").replace("\t", "").replace("\n", "")
            value = self._normalize_value(raw_value, validate)
            
            if value is not None:
                confidence = self._calculate_extraction_confidence(match.group(0), validate)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_value = value
                    best_match = match.group(0)
        
        if best_value is not None:
            return ExtractedField(best_value, best_confidence, best_match)
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
            elif validate == 'float':
                # Handle various number formats (with commas as decimal separators, etc.)
                cleaned = raw.replace(',', '.').replace(' ', '').replace('\u00A0', '')
                cleaned = re.sub(r'[^\d.-]', '', cleaned)
                if cleaned:
                    return float(cleaned)
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
                except ValueError:
                    pass
                    
        except Exception as e:
            logger.debug(f"Value normalization error for '{raw}' with validation '{validate}': {e}")
            pass
        return None

    def _extract_with_ner(self, text: str) -> Dict[str, ExtractedField]:
        fields = {}
        if not self.nlp:
            logger.debug("NER недоступен")
            return fields
        logger.debug("Запуск NER для извлечения сущностей")
        doc = self.nlp(text)
        for ent in doc.ents:
            logger.debug(f"Найдена сущность: {ent.text} -> {ent.label_}")
            if ent.label_ == "MONEY":
                # Assume MONEY is amount
                value = self._normalize_value(ent.text, 'float')
                if value is not None:
                    fields["amount"] = ExtractedField(value, 0.9, ent.text)
                    logger.debug(f"Извлечено MONEY: {value}")
            elif ent.label_ == "DATE":
                value = self._normalize_value(ent.text, 'date')
                if value is not None:
                    fields["date"] = ExtractedField(value, 0.9, ent.text)
                    logger.debug(f"Извлечено DATE: {value}")
            # Add more mappings as needed
        logger.debug(f"NER завершено: извлечено {len(fields)} полей")
        return fields

class SemanticParser:
    def __init__(self, config_manager: ConfigManager, table_builder: TableBuilder):
        self.field_extractor = FieldExtractor(config_manager)
        self.table_builder = table_builder
        self.document_parsers = {
            'invoice': self._parse_generic,
            'payment_order': self._parse_generic,
            'act': self._parse_generic,
            'invoice_factura': self._parse_generic,
            'dismissal_act': self._parse_generic,
            'reconciliation_act': self._parse_reconciliation_act
        }

    def parse_document(self, text: str, structure: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        parsed = self.document_parsers.get(doc_type, self._parse_generic)(text, structure)
        # Apply consistency validation
        if 'fields' in parsed:
            parsed['fields'] = self._validate_consistency(parsed['fields'], doc_type)
        return parsed

    def _parse_generic(self, text: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        fields = self.field_extractor.extract_fields(text, 'generic')
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
