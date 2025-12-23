import logging
import os
from pathlib import Path
import sys
import torch


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

backend_str = str(BACKEND_DIR)
if backend_str not in sys.path:
    sys.path.insert(0, backend_str)
    
from src.models.base_model import BasicNeuralClassifier

def load_best_checkpoint(model: BasicNeuralClassifier, 
                         optimizer: torch.optim.Optimizer, 
                         logger: logging.Logger, checkpoint: str, device="cpu"):
    if not os.path.exists(checkpoint):
        print(checkpoint)
        logger.error(f'Директория \'{checkpoint}\' не существует')
        raise

    checkpoint = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict']) # type: ignore
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # type: ignore

    logger.info(f"Загружена лучшая модель из чекпоинта")
    return model, optimizer