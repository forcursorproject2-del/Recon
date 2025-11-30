"""
Модуль валидации и безопасности для продакшена.

Обеспечивает:
- Валидацию путей файлов
- Безопасные файловые операции
- Валидацию входных данных
- Защиту от path traversal атак
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Исключение для ошибок валидации."""
    pass


class SecurityError(Exception):
    """Исключение для ошибок безопасности."""
    pass


def validate_file_path(file_path: str, base_dir: Optional[Path] = None, allowed_extensions: Optional[list] = None) -> Path:
    """
    Валидирует и нормализует путь к файлу.
    
    Args:
        file_path: Путь к файлу
        base_dir: Базовый директорий (для защиты от path traversal)
        allowed_extensions: Список разрешенных расширений (например, ['.pdf', '.jpg'])
    
    Returns:
        Нормализованный Path объект
        
    Raises:
        SecurityError: Если путь выходит за пределы base_dir
        ValidationError: Если расширение не разрешено
    """
    try:
        path = Path(file_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Некорректный путь к файлу: {file_path}") from e
    
    # Проверка на path traversal
    if base_dir is not None:
        base_dir = Path(base_dir).resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise SecurityError(f"Путь выходит за пределы базового директория: {file_path}")
    
    # Проверка расширения
    if allowed_extensions is not None:
        ext = path.suffix.lower()
        if ext not in allowed_extensions:
            raise ValidationError(f"Расширение файла '{ext}' не разрешено. Разрешенные: {allowed_extensions}")
    
    return path


def validate_folder_path(folder_path: str, must_exist: bool = True) -> Path:
    """
    Валидирует путь к папке.
    
    Args:
        folder_path: Путь к папке
        must_exist: Должна ли папка существовать
    
    Returns:
        Нормализованный Path объект
        
    Raises:
        ValidationError: Если папка не существует (если must_exist=True)
    """
    try:
        path = Path(folder_path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Некорректный путь к папке: {folder_path}") from e
    
    if must_exist and not path.exists():
        raise ValidationError(f"Папка не существует: {folder_path}")
    
    if not path.is_dir():
        raise ValidationError(f"Путь не является папкой: {folder_path}")
    
    return path


def validate_filename(filename: str, max_length: int = 255) -> str:
    """
    Валидирует имя файла.
    
    Args:
        filename: Имя файла
        max_length: Максимальная длина имени
    
    Returns:
        Валидированное имя файла
        
    Raises:
        ValidationError: Если имя файла некорректно
    """
    if not filename or not filename.strip():
        raise ValidationError("Имя файла не может быть пустым")
    
    if len(filename) > max_length:
        raise ValidationError(f"Имя файла слишком длинное (максимум {max_length} символов)")
    
    # Запрещенные символы в именах файлов
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    if re.search(invalid_chars, filename):
        raise ValidationError(f"Имя файла содержит недопустимые символы: {filename}")
    
    # Запрещенные имена (Windows)
    forbidden_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 
                       'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 'LPT4', 
                       'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
    name_without_ext = Path(filename).stem.upper()
    if name_without_ext in forbidden_names:
        raise ValidationError(f"Имя файла зарезервировано системой: {filename}")
    
    return filename.strip()


def safe_file_size(file_path: Path, max_size_mb: float = 100.0) -> Tuple[bool, float]:
    """
    Проверяет размер файла.
    
    Args:
        file_path: Путь к файлу
        max_size_mb: Максимальный размер в МБ
    
    Returns:
        Tuple (is_valid, size_mb)
    """
    if not file_path.exists():
        return False, 0.0
    
    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb <= max_size_mb, size_mb


def sanitize_filename(filename: str) -> str:
    """
    Очищает имя файла от опасных символов.
    
    Args:
        filename: Исходное имя файла
    
    Returns:
        Безопасное имя файла
    """
    # Убираем путь, оставляем только имя
    safe_name = Path(filename).name
    
    # Заменяем опасные символы на подчеркивания
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', safe_name)
    
    # Убираем ведущие и завершающие точки и пробелы
    safe_name = safe_name.strip('. ')
    
    # Ограничиваем длину
    if len(safe_name) > 200:
        name_part = safe_name[:190]
        ext = Path(safe_name).suffix
        safe_name = name_part + ext
    
    return safe_name if safe_name else "unnamed_file"


def validate_json_structure(data: dict, required_keys: list, path: str = "root") -> None:
    """
    Валидирует структуру JSON данных.
    
    Args:
        data: Словарь для проверки
        required_keys: Список обязательных ключей
        path: Путь в структуре (для сообщений об ошибках)
    
    Raises:
        ValidationError: Если структура некорректна
    """
    if not isinstance(data, dict):
        raise ValidationError(f"Ожидается словарь в '{path}', получен {type(data).__name__}")
    
    for key in required_keys:
        if key not in data:
            raise ValidationError(f"Отсутствует обязательный ключ '{key}' в '{path}'")

