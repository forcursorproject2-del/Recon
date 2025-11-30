"""
Key-Value парсер для актов сверки.

Принцип работы: ищет ключевые якоря в тексте и извлекает значения рядом с ними.
Работает даже на "убитых" сканах, так как ключи-якоря обычно читаются хорошо.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class KeyValueReconciliationParser:
    """
    Парсер актов сверки на основе key-value подхода.
    
    Ищет только ключевые поля:
    - Номер и дата акта
    - Наименование контрагента
    - ИНН контрагента
    - Конечное сальдо
    - Направление долга (мы должны / нам должны)
    """
    
    def __init__(self, user_org_name: str = "", user_org_inn: str = ""):
        """
        Args:
            user_org_name: Название нашей организации
            user_org_inn: ИНН нашей организации
        """
        self.user_org_name = user_org_name.upper() if user_org_name else ""
        self.user_org_inn = user_org_inn
        
        # Паттерны для поиска номера акта (с учетом ошибок OCR)
        self.act_number_patterns = [
            r'[№NН]\s*(?:ЦБ|ЛБ|ТЗ|АС|АСВ|цб|лб|тз|ас)[\s\-]*(\d+)',
            r'[№NН]\s*(\d{4,})',  # Номер из 4+ цифр
            r'номер[:\s]+(\d{4,})',
            r'[№NН][oо]?\s*(\d{4,})',  # С учетом ошибок OCR "No" вместо "№"
        ]
        
        # Паттерны для поиска даты (с учетом ошибок OCR)
        self.date_patterns = [
            r'[оО][тT]\s+(\d{1,2})[\.\s]+(\d{1,2})[\.\s]+(\d{4})',
            r'(\d{1,2})[\.\s]+(\d{1,2})[\.\s]+(\d{4})\s+[гГ]\.?',
            r'(\d{1,2})\s+(?:январ[ья]|феврал[ья]|март[а]|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+(\d{4})',
            r'(\d{1,2})[\.\s]+(?:январ[ья]|феврал[ья]|март[а]|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+(\d{4})',
        ]
        
        # Ключевые слова для поиска организаций (с учетом ошибок OCR)
        self.org_keywords = [
            r'мы[,]?\s+нижеподписавшиеся',
            r'нижеп[дД]лисашиса',  # Ошибка OCR
            r'с\s+одной\s+стороны',
            r'[дД]ной\s+[cС]tupolel',  # Ошибка OCR
            r'организация',
            r'ооо\s*[«"]',
            r'ао\s*[«"]',
            r'ип\s*[«"]',
            r'с\s+другой\s+стороны',
            r'другой\s+[cС]t[оО]роны',  # Ошибка OCR
            r'в\s+лице',
            r'взаимных\s+расчетов\s+с',
            r'[зЗ]аимных\s+расчетов',  # Ошибка OCR
            r'акционерное\s+об[шШ]ество',  # Ошибка OCR "обшстао"
            r'акционерное\s+об[цЦ][eЕ][сС][тT]',  # Ошибка OCR "обцест"
        ]
        
        # Ключевые слова для конечного сальдо
        self.balance_keywords = [
            r'исходящее\s+сальдо',
            r'конечное\s+сальдо',
            r'на\s+\d{1,2}\.\d{1,2}\.\d{4}\s+остаток',
            r'итого\s+задолженность',
            r'остаток\s+на\s+конец',
            r'всего\s+по\s+акту',
            r'итого\s+по\s+акту',
        ]
    
    def parse(self, text: str, blocks: Optional[List] = None) -> Dict[str, any]:
        """
        Парсит акт сверки и извлекает ключевые поля.
        
        Args:
            text: Распознанный текст документа
            blocks: Блоки PaddleOCR (опционально, для будущего использования)
            
        Returns:
            Словарь с извлеченными полями и confidence
        """
        if not text:
            return self._empty_result()
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return self._empty_result()
        
        result = {
            "act_number": None,
            "act_date": None,
            "counterparty": None,
            "counterparty_inn": None,
            "final_balance": 0.0,
            "we_owe": True,  # True = мы должны, False = нам должны
            "confidence": 0.0,
            "raw_text": text[:500]  # Первые 500 символов для отладки
        }
        
        # 1. Номер и дата акта (обычно вверху документа)
        act_info = self._extract_act_number_and_date(lines[:20])
        result["act_number"] = act_info["number"]
        result["act_date"] = act_info["date"]
        
        # 2. Наименование контрагента
        result["counterparty"] = self._extract_counterparty(text, lines)
        
        # 3. ИНН контрагента
        result["counterparty_inn"] = self._extract_counterparty_inn(text)
        
        # 4. Конечное сальдо (ищем в нижней трети документа)
        balance_info = self._extract_final_balance(lines)
        result["final_balance"] = balance_info["amount"]
        
        # 5. Направление долга
        result["we_owe"] = self._determine_debt_direction(text, result["counterparty"])
        
        # 6. Расчет confidence
        result["confidence"] = self._calculate_confidence(result)
        
        logger.debug(f"Key-Value парсинг: confidence={result['confidence']:.2f}, "
                    f"act_number={result['act_number']}, "
                    f"counterparty={result['counterparty'][:30] if result['counterparty'] else None}, "
                    f"balance={result['final_balance']}")
        
        return result
    
    def _empty_result(self) -> Dict:
        """Возвращает пустой результат."""
        return {
            "act_number": None,
            "act_date": None,
            "counterparty": None,
            "counterparty_inn": None,
            "final_balance": 0.0,
            "we_owe": True,
            "confidence": 0.0,
            "raw_text": ""
        }
    
    def _extract_act_number_and_date(self, lines: List[str]) -> Dict[str, Optional[str]]:
        """Извлекает номер и дату акта из первых строк."""
        result = {"number": None, "date": None}
        
        # Объединяем первые строки для поиска
        header_text = " ".join(lines[:5])
        
        # Поиск номера акта
        for pattern in self.act_number_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                result["number"] = match.group(1)
                break
        
        # Поиск даты
        for pattern in self.date_patterns:
            match = re.search(pattern, header_text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 3:
                        day, month, year = match.groups()
                        # Нормализуем дату
                        day = day.zfill(2)
                        month = month.zfill(2)
                        result["date"] = f"{day}.{month}.{year}"
                        break
                except Exception as e:
                    logger.debug(f"Ошибка парсинга даты: {e}")
                    continue
        
        return result
    
    def _extract_counterparty(self, text: str, lines: List[str]) -> Optional[str]:
        """Извлекает наименование контрагента."""
        # Ищем организации в тексте
        orgs = self._find_organizations(text, lines)
        
        if len(orgs) >= 2:
            # Если нашли две организации, выбираем ту, которая не наша
            for org in orgs:
                if self.user_org_name and self.user_org_name.upper() in org.upper():
                    continue
                return org
            # Если не нашли нашу, возвращаем вторую
            return orgs[1] if len(orgs) > 1 else orgs[0]
        elif len(orgs) == 1:
            # Если одна организация и она не наша - возвращаем её
            if not self.user_org_name or self.user_org_name.upper() not in orgs[0].upper():
                return orgs[0]
        
        return None
    
    def _find_organizations(self, text: str, lines: List[str]) -> List[str]:
        """Находит наименования организаций в тексте."""
        orgs = []
        
        # Паттерны для поиска организаций (с учетом ошибок OCR)
        org_patterns = [
            r'(?:ООО|АО|ИП|ПАО|ЗАО|ОБЩЕСТВО|об[шШ]ество|об[цЦ][eЕ][сС][тT])\s*[«"]([^«"]+)[»"]',
            r'(?:ООО|АО|ИП|ПАО|ЗАО|ОБЩЕСТВО|об[шШ]ество|об[цЦ][eЕ][сС][тT])\s+([А-ЯЁ][А-ЯЁа-яё\s]+?)(?:\s+ИНН|\s+г\.|\s+адрес|$)',
            r'[«"]([А-ЯЁ][А-ЯЁа-яё\s]{5,30})[»"]',
            r'акционерное\s+(?:об[шШ]ество|об[цЦ][eЕ][сС][тT])\s+([А-ЯЁа-яё\s]{5,50}?)(?:\s+ИНН|\s+г\.|\s+адрес|$)',
            r'([А-ЯЁ][а-яё]{3,}\s+[А-ЯЁ][а-яё]{3,}\s+(?:комбинат|завод|предприятие|компания))',  # Например "Барнаульский молочный комбинат"
        ]
        
        for pattern in org_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                org_name = match.group(1).strip()
                # Фильтруем слишком короткие или длинные
                if 5 <= len(org_name) <= 100:
                    # Убираем дубликаты
                    if org_name not in orgs:
                        orgs.append(org_name)
        
        # Также ищем после ключевых слов
        for keyword in self.org_keywords:
            pattern = rf'{keyword}[:\s]+([А-ЯЁ][А-ЯЁа-яё\s]{5,50}?)(?:\s+ИНН|\s+г\.|\s+адрес|$)'
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                org_name = match.group(1).strip()
                if 5 <= len(org_name) <= 100 and org_name not in orgs:
                    orgs.append(org_name)
        
        return orgs[:3]  # Возвращаем максимум 3 организации
    
    def _extract_counterparty_inn(self, text: str) -> Optional[str]:
        """Извлекает ИНН контрагента (второй найденный ИНН, если первый - наш)."""
        # Ищем все ИНН в тексте
        inn_pattern = r'ИНН[\s:]*(\d{10,12})'
        inns = re.findall(inn_pattern, text, re.IGNORECASE)
        
        if not inns:
            return None
        
        # Если указан наш ИНН и он первый - возвращаем второй
        if self.user_org_inn and len(inns) >= 2:
            if inns[0] == self.user_org_inn:
                return inns[1]
            else:
                return inns[0]
        
        # Если наш ИНН не указан или не найден - возвращаем первый
        return inns[0] if inns else None
    
    def _extract_final_balance(self, lines: List[str]) -> Dict[str, float]:
        """Извлекает конечное сальдо из нижней части документа."""
        result = {"amount": 0.0}
        
        # Берем последние 50 строк (нижняя треть документа)
        lower_lines = lines[-50:] if len(lines) > 50 else lines
        lower_text = "\n".join(lower_lines)
        
        # Ищем ключевые слова для сальдо
        for keyword in self.balance_keywords:
            pattern = rf'{keyword}[\s:]*([\d\s,\.]+)'
            matches = re.finditer(pattern, lower_text, re.IGNORECASE)
            
            for match in matches:
                amount_str = match.group(1).strip()
                # Нормализуем число (убираем пробелы, заменяем запятую на точку)
                amount_str = amount_str.replace(' ', '').replace(',', '.')
                try:
                    amount = float(amount_str)
                    if amount > result["amount"]:
                        result["amount"] = amount
                except ValueError:
                    continue
        
        # Если не нашли по ключевым словам, ищем самую большую цифру в нижней части
        if result["amount"] == 0.0:
            # Ищем все числа в нижней части (с учетом ошибок OCR)
            # Паттерн для чисел с пробелами, запятыми, точками
            numbers = re.findall(r'[\d\s,\.]+', lower_text)
            max_amount = 0.0
            for num_str in numbers:
                try:
                    # Нормализуем: убираем пробелы, заменяем запятую на точку
                    num_str_clean = num_str.replace(' ', '').replace(',', '.')
                    # Убираем лишние точки (оставляем только одну)
                    if num_str_clean.count('.') > 1:
                        parts = num_str_clean.split('.')
                        num_str_clean = parts[0] + '.' + ''.join(parts[1:])
                    num = float(num_str_clean)
                    # Игнорируем маленькие числа (даты, номера) и очень большие (вероятно ошибки)
                    if 1000 <= num <= 1000000000 and num > max_amount:
                        max_amount = num
                except ValueError:
                    continue
            result["amount"] = max_amount
        
        return result
    
    def _determine_debt_direction(self, text: str, counterparty: Optional[str]) -> bool:
        """
        Определяет направление долга.
        
        Returns:
            True - мы должны контрагенту
            False - нам должны (переплата)
        """
        if not self.user_org_name:
            return True  # По умолчанию считаем, что мы должны
        
        # Ищем фразу "по данным"
        pattern = r'по\s+данным\s+([^\.]+)'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            org_mentioned = match.group(1).strip()
            # Проверяем, упоминается ли наша организация
            if self.user_org_name.upper() in org_mentioned.upper():
                return True  # По данным нашей организации → мы должны
            # Проверяем, упоминается ли контрагент
            if counterparty and counterparty.upper()[:15] in org_mentioned.upper():
                return False  # По данным контрагента → нам должны
        
        # Если не нашли "по данным", проверяем контекст вокруг сальдо
        # Обычно если сальдо в дебете - мы должны, если в кредите - нам должны
        if re.search(r'дебет|долг|задолженность', text, re.IGNORECASE):
            return True
        
        return True  # По умолчанию
    
    def _calculate_confidence(self, result: Dict) -> float:
        """
        Рассчитывает confidence на основе найденных полей.
        
        Каждое поле добавляет к confidence:
        - Номер акта: +0.15
        - Дата акта: +0.15
        - Контрагент: +0.25
        - ИНН: +0.20
        - Сальдо: +0.25
        """
        confidence = 0.0
        
        if result["act_number"]:
            confidence += 0.15
        if result["act_date"]:
            confidence += 0.15
        if result["counterparty"]:
            confidence += 0.25
        if result["counterparty_inn"]:
            confidence += 0.20
        if result["final_balance"] > 0:
            confidence += 0.25
        
        return min(confidence, 1.0)
    
    def format_result(self, result: Dict) -> str:
        """
        Форматирует результат для вывода пользователю.
        
        Args:
            result: Результат парсинга
            
        Returns:
            Отформатированная строка
        """
        lines = []
        
        # Номер и дата акта
        if result["act_number"] and result["act_date"]:
            lines.append(f"Акт сверки № {result['act_number']} от {result['act_date']}")
        elif result["act_date"]:
            lines.append(f"Акт сверки от {result['act_date']}")
        elif result["act_number"]:
            lines.append(f"Акт сверки № {result['act_number']}")
        else:
            lines.append("Акт сверки")
        
        # Контрагент
        if result["counterparty"]:
            lines.append(f"Контрагент: {result['counterparty']}")
        
        # ИНН
        if result["counterparty_inn"]:
            lines.append(f"ИНН: {result['counterparty_inn']}")
        
        # Направление долга
        if result["counterparty"]:
            if result["we_owe"]:
                lines.append(f"По данным {result['counterparty']} (наша организация)")
            else:
                lines.append(f"По данным {result['counterparty']} (контрагент)")
        
        # Сальдо
        if result["final_balance"] > 0:
            balance_str = f"{result['final_balance']:,.2f}".replace(',', ' ').replace('.', ',')
            if result["we_owe"]:
                lines.append(f"Остаток: {balance_str} руб.")
                lines.append(f"Вы должны поставщику {balance_str} руб.")
            else:
                lines.append(f"Остаток: {balance_str} руб.")
                lines.append(f"Вам должны {balance_str} руб.")
        
        return "\n".join(lines)

