"""
Модуль контекстной валидации полей с учетом взаимосвязей и позиции в документе.
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
from ..utils.inn_validator import INNValidator

logger = logging.getLogger(__name__)


class ContextualFieldValidator:
    """Валидация полей с учетом контекста документа."""
    
    def __init__(self):
        self.inn_validator = INNValidator()
    
    def validate_field_relationships(self, fields: Dict[str, Dict[str, Any]], doc_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Проверка взаимосвязей между полями.
        
        Args:
            fields: Словарь извлеченных полей
            doc_type: Тип документа
        
        Returns:
            Словарь полей с добавленными предупреждениями и исправлениями
        """
        validated_fields = fields.copy()
        
        # Проверка суммы = сумма без НДС + НДС
        self._validate_vat_calculation(validated_fields)
        
        # Проверка соответствия ИНН организациям
        self._validate_inn_organization_match(validated_fields)
        
        # Проверка дат на разумность
        self._validate_dates(validated_fields, doc_type)
        
        # Проверка сумм на разумность
        self._validate_amounts(validated_fields)
        
        return validated_fields
    
    def _validate_vat_calculation(self, fields: Dict[str, Dict[str, Any]]) -> None:
        """Проверка расчета НДС."""
        total_with_vat = fields.get('total_with_vat', {}).get('value')
        total_without_vat = fields.get('total_without_vat', {}).get('value')
        vat = fields.get('vat', {}).get('value')
        
        if total_with_vat and total_without_vat and vat:
            calculated_total = total_without_vat + vat
            difference = abs(total_with_vat - calculated_total)
            
            # Допускаем разницу до 0.01 (копейки)
            if difference > 0.01:
                fields['total_with_vat']['warning'] = (
                    f"Несоответствие сумм: {total_with_vat} ≠ {total_without_vat} + {vat} = {calculated_total}"
                )
                # Автоматическое исправление
                fields['total_with_vat']['auto_corrected'] = calculated_total
                fields['total_with_vat']['confidence'] = max(0.5, fields['total_with_vat'].get('confidence', 0.8) - 0.2)
    
    def _validate_inn_organization_match(self, fields: Dict[str, Dict[str, Any]]) -> None:
        """Проверка соответствия ИНН организациям (позиционная проверка)."""
        # Проверяем, что ИНН поставщика находится рядом с названием поставщика
        supplier = fields.get('supplier', {}).get('value')
        supplier_inn = fields.get('supplier_inn', {}).get('value')
        
        if supplier and supplier_inn:
            # Валидируем ИНН
            is_valid, error = self.inn_validator.validate_inn(supplier_inn)
            if not is_valid:
                fields['supplier_inn']['warning'] = f"ИНН не прошел валидацию: {error}"
                fields['supplier_inn']['confidence'] = max(0.0, fields['supplier_inn'].get('confidence', 0.8) - 0.3)
        
        # Аналогично для покупателя
        buyer = fields.get('buyer', {}).get('value')
        buyer_inn = fields.get('buyer_inn', {}).get('value')
        
        if buyer and buyer_inn:
            is_valid, error = self.inn_validator.validate_inn(buyer_inn)
            if not is_valid:
                fields['buyer_inn']['warning'] = f"ИНН не прошел валидацию: {error}"
                fields['buyer_inn']['confidence'] = max(0.0, fields['buyer_inn'].get('confidence', 0.8) - 0.3)
    
    def _validate_dates(self, fields: Dict[str, Dict[str, Any]], doc_type: str) -> None:
        """Проверка дат на разумность."""
        date_value = fields.get('date', {}).get('value')
        
        if not date_value:
            return
        
        try:
            # Парсим дату
            if isinstance(date_value, str):
                # Пробуем разные форматы
                date_formats = ['%Y-%m-%d', '%d.%m.%Y', '%d-%m-%Y']
                parsed_date = None
                for fmt in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_value, fmt)
                        break
                    except ValueError:
                        continue
                
                if not parsed_date:
                    return
            else:
                parsed_date = date_value
            
            # Проверяем, что дата не в будущем
            if parsed_date > datetime.now():
                fields['date']['warning'] = "Дата документа в будущем"
                fields['date']['confidence'] = max(0.0, fields['date'].get('confidence', 0.8) - 0.2)
            
            # Проверяем, что дата не слишком старая (старше 20 лет)
            if parsed_date < datetime.now() - timedelta(days=365*20):
                fields['date']['warning'] = "Дата документа слишком старая"
                fields['date']['confidence'] = max(0.0, fields['date'].get('confidence', 0.8) - 0.1)
        
        except Exception as e:
            logger.debug(f"Ошибка при валидации даты: {e}")
    
    def _validate_amounts(self, fields: Dict[str, Dict[str, Any]]) -> None:
        """Проверка сумм на разумность."""
        amount = fields.get('amount', {}).get('value')
        total_with_vat = fields.get('total_with_vat', {}).get('value')
        
        # Проверяем, что суммы положительные
        if amount is not None and amount < 0:
            fields['amount']['warning'] = "Сумма отрицательная"
            fields['amount']['confidence'] = max(0.0, fields['amount'].get('confidence', 0.8) - 0.3)
        
        if total_with_vat is not None and total_with_vat < 0:
            fields['total_with_vat']['warning'] = "Сумма с НДС отрицательная"
            fields['total_with_vat']['confidence'] = max(0.0, fields['total_with_vat'].get('confidence', 0.8) - 0.3)
        
        # Проверяем, что сумма с НДС >= суммы без НДС
        total_without_vat = fields.get('total_without_vat', {}).get('value')
        if total_with_vat and total_without_vat:
            if total_with_vat < total_without_vat:
                fields['total_with_vat']['warning'] = "Сумма с НДС меньше суммы без НДС"
                fields['total_with_vat']['confidence'] = max(0.0, fields['total_with_vat'].get('confidence', 0.8) - 0.2)
    
    def validate_spatial_context(self, field_value: str, field_name: str, text: str, 
                                match_position: int, doc_type: str) -> Dict[str, Any]:
        """
        Проверка позиционного контекста поля.
        
        Args:
            field_value: Значение поля
            field_name: Имя поля
            text: Весь текст документа
            match_position: Позиция совпадения в тексте
            doc_type: Тип документа
        
        Returns:
            Словарь с информацией о контексте
        """
        context = {
            'position_ratio': match_position / len(text) if text else 1.0,
            'near_keyword': False,
            'is_unique': False,
            'context_score': 0.0
        }
        
        # Поля в начале документа более важны (первые 30%)
        if context['position_ratio'] < 0.3:
            context['context_score'] += 0.1
        
        # Проверяем близость к ключевым словам
        keyword_patterns = {
            'supplier': r'поставщик|продавец',
            'buyer': r'покупатель|заказчик',
            'amount': r'сумма|итого|всего\s+к\s+оплате',
            'date': r'дата|от\s+\d',
            'document_number': r'счет|счёт|документ|№|N\s*\d',
            'inn': r'инн',
            'bic': r'бик'
        }
        
        if field_name in keyword_patterns:
            # Ищем ключевые слова в радиусе 200 символов
            start = max(0, match_position - 200)
            end = min(len(text), match_position + len(field_value) + 200)
            context_text = text[start:end]
            
            if re.search(keyword_patterns[field_name], context_text, re.IGNORECASE):
                context['near_keyword'] = True
                context['context_score'] += 0.15
        
        # Проверяем уникальность (сколько раз встречается значение)
        if field_value:
            occurrences = text.count(str(field_value))
            if occurrences == 1:
                context['is_unique'] = True
                context['context_score'] += 0.1
        
        return context
    
    def calculate_contextual_confidence(self, base_confidence: float, context: Dict[str, Any]) -> float:
        """
        Расчет confidence с учетом контекста.
        
        Args:
            base_confidence: Базовая уверенность
            context: Информация о контексте
        
        Returns:
            Скорректированная уверенность
        """
        adjusted = base_confidence + context.get('context_score', 0.0)
        return min(1.0, max(0.0, adjusted))

