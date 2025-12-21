from typing import Dict, Type
from src.core.base_classes import DataModule, SystemPipeline
from src.models.base_model import BasicNeuralClassifier

MODEL_REGISTRY: Dict[str, Type[BasicNeuralClassifier]] = {}
DATAMODULE_REGISTRY: Dict[str, Type[DataModule]] = {}
PIPELINE_REGISTRY: Dict[str, Type[SystemPipeline]] = {}

def register_model(name: str):
    """Декоратор для регистрации модели"""
    def decorator(cls: Type[BasicNeuralClassifier]) -> Type[BasicNeuralClassifier]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def register_datamodule(name: str):
    """Декоратор для регистрации датамодуля"""
    def decorator(cls: Type[DataModule]) -> Type[DataModule]:
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator

def register_pipeline(name: str):
    """Декоратор для регистрации датасета"""
    def decorator(cls: Type[SystemPipeline]) -> Type[SystemPipeline]:
        PIPELINE_REGISTRY[name] = cls
        return cls
    return decorator
        