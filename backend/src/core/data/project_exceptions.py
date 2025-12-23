from typing import Any


class DataError(Exception):
    """Базовое для ошибок с данными"""
    pass

class LengthMismatch(DataError):
    """Ошибка соответствия по количеству в GlossingPipeline"""
    def __init__(self, text: str, sent: str, value_one: Any, value_two: Any):
        super().__init__(f"Количество слов в тексте {text} в предложении {sent} не совпадает с данными: {len(value_one)} vs {len(value_two)}")

class DataMissing(DataError):
    """Ошибка отсутствия данных"""
    def __init__(self):
        super().__init__(f"Данные не были предоставлены")

class ComponentError(Exception):
    """Базовое для ошибок компонентов"""

class ComponentInitializationError(ComponentError):
    """Ошибка инициализации компонента"""
    def __init__(self, component_name: str, reason: str):
        self.component_name = component_name
        self.reason = reason
        super().__init__(f"Не получилось инициализировать модель {component_name}: {reason}")