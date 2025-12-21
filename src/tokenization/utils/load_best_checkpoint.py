import logging
import os
import torch
from src.models.base_model import BasicNeuralClassifier

def load_best_checkpoint(model: BasicNeuralClassifier, 
                         optimizer: torch.optim.Optimizer, 
                         logger: logging.Logger, checkpoint: str, device="cpu"):
    if not os.path.exists(checkpoint):
        logger.error(f'Директория \'{checkpoint}\' не существует')
        raise

    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict']) # type: ignore
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # type: ignore

    logger.info(f"Загружена лучшая модель из чекпоинта")
    return model, optimizer