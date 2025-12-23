from typing import Any


class DataError(Exception):
    pass

class LengthMismatch(DataError):
    """Ошибка соответствия по количеству в GlossingPipeline"""
    def __init__(self, text: str, sent: str, value_one: Any, value_two: Any):
        super().__init__(f"Количество слов в тексте {text} в предложении {sent} не совпадает с данными: {len(value_one)} vs {len(value_two)}")

class DataIsMissing(DataError):
    """Ошибка отсутствия данных"""
    def __init__(self):
        super().__init__(f"Данные не были предоставлены")


class ComponentError(Exception):
    """Базовое для ошибок компонентов"""
    pass

class ComponentNotRegisteredError(ComponentError):
    """Компонент не зарегистрирован"""
    def __init__(self, component_type: str, name: str):
        self.component_type = component_type
        self.name = name
        super().__init__(f"{component_type} '{name}' is not registered")

class ComponentInitializationError(ComponentError):
    """Ошибка инициализации компонента"""
    def __init__(self, component_name: str, reason: str):
        self.component_name = component_name
        self.reason = reason
        super().__init__(f"Не получилось инициализировать модель {component_name}: {reason}")