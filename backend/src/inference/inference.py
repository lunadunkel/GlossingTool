from typing import Type, Any

from src.core.base_classes import BaseInference
from src.inference.inference_models import *
from src.core.initializing import INFERENCE_REGISTRY


class Inference:
    def __init__(self, config, model, logger, device='cpu'):
        self.task = config.task
        inference_cls: Type[BaseInference] = INFERENCE_REGISTRY[self.task]
        self._inference_instance: BaseInference = inference_cls(config=config, model=model, # type: ignore
                                                                logger=logger, device=device)  # type: ignore

    def predict(self, data):
        return self._inference_instance.predict(data)
