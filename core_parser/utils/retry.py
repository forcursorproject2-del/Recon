"""
Модуль для повторных попыток выполнения операций.

Обеспечивает надежность в продакшене через механизм retry.
"""

import time
import logging
from typing import Callable, TypeVar, Any, Optional, List
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Декоратор для повторных попыток выполнения функции.
    
    Args:
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками в секундах
        backoff: Множитель для экспоненциальной задержки
        exceptions: Кортеж исключений, при которых следует повторять попытки
        on_failure: Функция обратного вызова при неудаче (принимает exception и attempt_number)
    
    Returns:
        Декорированная функция
    
    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(ValueError, IOError))
        def risky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Попытка {attempt}/{max_attempts} не удалась для {func.__name__}: {e}. "
                            f"Повтор через {current_delay:.2f} сек..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"Все {max_attempts} попыток не удались для {func.__name__}: {e}"
                        )
                        if on_failure:
                            try:
                                on_failure(e, attempt)
                            except Exception as callback_error:
                                logger.error(f"Ошибка в callback on_failure: {callback_error}")
            
            # Все попытки исчерпаны
            if on_failure:
                try:
                    on_failure(last_exception, max_attempts)
                except Exception as callback_error:
                    logger.error(f"Ошибка в финальном callback on_failure: {callback_error}")
            
            raise last_exception
        
        return wrapper
    return decorator


class RetryableOperation:
    """
    Класс для выполнения операций с повторными попытками.
    
    Полезен когда нужно более гибкое управление retry логикой.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.exceptions = exceptions
    
    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Выполняет функцию с повторными попытками.
        
        Args:
            func: Функция для выполнения
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы
        
        Returns:
            Результат выполнения функции
        
        Raises:
            Последнее исключение, если все попытки не удались
        """
        last_exception = None
        current_delay = self.delay
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                last_exception = e
                
                if attempt < self.max_attempts:
                    logger.warning(
                        f"Попытка {attempt}/{self.max_attempts} не удалась: {e}. "
                        f"Повтор через {current_delay:.2f} сек..."
                    )
                    time.sleep(current_delay)
                    current_delay *= self.backoff
                else:
                    logger.error(f"Все {self.max_attempts} попыток не удались: {e}")
        
        raise last_exception

