"""
Модуль валидации ИНН с проверкой контрольной суммы.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class INNValidator:
    """Класс для валидации ИНН с проверкой контрольной суммы."""
    
    @staticmethod
    def validate_inn(inn: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация ИНН с проверкой контрольной суммы.
        
        Args:
            inn: ИНН для проверки (строка из 10 или 12 цифр)
        
        Returns:
            Tuple (is_valid, error_message)
            is_valid: True если ИНН валиден
            error_message: Сообщение об ошибке или None
        """
        if not inn:
            return False, "ИНН не может быть пустым"
        
        # Убираем все нецифровые символы
        inn_clean = re.sub(r'\D', '', str(inn))
        
        if len(inn_clean) == 10:
            return INNValidator._validate_inn_10(inn_clean)
        elif len(inn_clean) == 12:
            return INNValidator._validate_inn_12(inn_clean)
        else:
            return False, f"ИНН должен содержать 10 или 12 цифр, получено {len(inn_clean)}"
    
    @staticmethod
    def _validate_inn_10(inn: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация 10-значного ИНН (для юридических лиц).
        
        Args:
            inn: 10-значный ИНН
        
        Returns:
            Tuple (is_valid, error_message)
        """
        if len(inn) != 10:
            return False, "ИНН должен содержать 10 цифр"
        
        if not inn.isdigit():
            return False, "ИНН должен содержать только цифры"
        
        # Коэффициенты для расчета контрольной суммы
        coefficients = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        
        # Вычисляем контрольную сумму
        checksum = sum(int(inn[i]) * coefficients[i] for i in range(9)) % 11
        
        # Если остаток равен 10, контрольная сумма = 0
        if checksum == 10:
            checksum = 0
        
        # Проверяем контрольную цифру
        if checksum != int(inn[9]):
            return False, f"Неверная контрольная сумма ИНН. Ожидалось {checksum}, получено {inn[9]}"
        
        return True, None
    
    @staticmethod
    def _validate_inn_12(inn: str) -> Tuple[bool, Optional[str]]:
        """
        Валидация 12-значного ИНН (для физических лиц и ИП).
        
        Args:
            inn: 12-значный ИНН
        
        Returns:
            Tuple (is_valid, error_message)
        """
        if len(inn) != 12:
            return False, "ИНН должен содержать 12 цифр"
        
        if not inn.isdigit():
            return False, "ИНН должен содержать только цифры"
        
        # Коэффициенты для первой контрольной суммы
        coefficients1 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        
        # Вычисляем первую контрольную сумму
        checksum1 = sum(int(inn[i]) * coefficients1[i] for i in range(10)) % 11
        
        # Если остаток равен 10, контрольная сумма = 0
        if checksum1 == 10:
            checksum1 = 0
        
        # Проверяем первую контрольную цифру
        if checksum1 != int(inn[10]):
            return False, f"Неверная первая контрольная сумма ИНН. Ожидалось {checksum1}, получено {inn[10]}"
        
        # Коэффициенты для второй контрольной суммы
        coefficients2 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        
        # Вычисляем вторую контрольную сумму
        checksum2 = sum(int(inn[i]) * coefficients2[i] for i in range(11)) % 11
        
        # Если остаток равен 10, контрольная сумма = 0
        if checksum2 == 10:
            checksum2 = 0
        
        # Проверяем вторую контрольную цифру
        if checksum2 != int(inn[11]):
            return False, f"Неверная вторая контрольная сумма ИНН. Ожидалось {checksum2}, получено {inn[11]}"
        
        return True, None
    
    @staticmethod
    def validate_and_adjust_confidence(inn: str, base_confidence: float) -> Tuple[Optional[str], float]:
        """
        Валидирует ИНН и корректирует confidence.
        
        Args:
            inn: ИНН для проверки
            base_confidence: Базовая уверенность
        
        Returns:
            Tuple (validated_inn, adjusted_confidence)
            validated_inn: Валидированный ИНН или None
            adjusted_confidence: Скорректированная уверенность
        """
        is_valid, error = INNValidator.validate_inn(inn)
        
        if is_valid:
            # Если ИНН валиден, увеличиваем confidence
            adjusted_confidence = min(1.0, base_confidence + 0.1)
            return inn, adjusted_confidence
        else:
            logger.debug(f"ИНН не прошел валидацию: {error}")
            # Если ИНН невалиден, снижаем confidence
            adjusted_confidence = max(0.0, base_confidence - 0.2)
            return None, adjusted_confidence

