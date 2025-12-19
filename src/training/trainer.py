"""
Скрипт обучения модели

Использование:
    python scripts/train.py --model model --config configs/model.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import random
import numpy as np
from src.training.utils.logger import setup_logger
from src.core.config import ExperimentConfig
from src.training.utils.seed import set_seed
from src.core.data.datamodules import *
from src.models.pos_tagger import PosTagger
from src.models.lemma_tagger import LemmaAffixTagger
from src.models.segmentation import MorphSegmentationCNN
from src.training.utils.load_data import clone_and_load_data
from src.training.utils.initializing import MODEL_REGISTRY, DATAMODULE_REGISTRY

# from src.models.pos_tagger import create_pos_tagger
# from src.training.trainers import PosTaggerTrainer
# from src.data.datamodules import PoSDataModule
# from src.data.readers import load_and_split_data


def main():
    parser = argparse.ArgumentParser(description="Train PoS-tagger model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to data"
    )
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = ExperimentConfig.from_yaml(args.config)
    model_name = args.model
    
    if args.device:
        config.model.device = args.device
        config.training.device = args.device
    
    if torch.cuda.is_available() and config.training.device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    
    config.model.device = device
    
    logger = setup_logger(
        config.name,
        log_dir=config.training.log_dir
    )
    
    logger.info(f"="*80)
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Device: {device}")
    logger.info(f"="*80)
    
    set_seed(config.training.random_seed)
    logger.info(f"Random seed: {config.training.random_seed}")
    
    logger.info("Loading data...")
    clone = True if args.data_path is not None else False
    train_data, val_data, test_data = clone_and_load_data(config=config.data,
                                                          clone=clone,
                                                          data_path=args.data_path)
    
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Val size: {len(val_data)}")
    logger.info(f"Test size: {len(test_data)}")
    
    
    logger.info("Creating data loaders...")

    if model_name not in DATAMODULE_REGISTRY:
        raise ValueError(f"Неизвестная модель: {model_name}")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {model_name}")

    model = MODEL_REGISTRY[model_name]


    # datamodule = 
#     datamodule = PoSDataModule(
#         train_data=train_data,
#         val_data=val_data,
#         test_data=test_data,
#         batch_size=config.training.batch_size,
#         tokenizer_config=config.tokenizer,
#         use_char_emb=config.model.use_char_emb
#     )
    
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()
#     test_loader = datamodule.test_dataloader()
    
#     logger.info(f"Train batches: {len(train_loader)}")
#     logger.info(f"Val batches: {len(val_loader)}")
#     logger.info(f"Test batches: {len(test_loader)}")
    
#     logger.info("Creating model...")
#     model = create_pos_tagger(config.model)
    
#     num_params = sum(p.numel() for p in model.parameters())
#     num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"Total parameters: {num_params:,}")
#     logger.info(f"Trainable parameters: {num_trainable:,}")
    
#     # Создание Trainer
#     logger.info("Creating trainer...")
#     trainer = PosTaggerTrainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         config=config.training,
#         use_char_emb=config.model.use_char_emb,
#         logger=logger
#     )
    
#     # Обучение
#     logger.info("Starting training...")
#     history = trainer.train()
    
#     # Сохранение истории
#     history_path = os.path.join(
#         config.training.log_dir,
#         f"{config.name}_history.json"
#     )
#     import json
#     with open(history_path, 'w') as f:
#         json.dump(history, f, indent=2)
#     logger.info(f"Training history saved to: {history_path}")
    
#     # Загрузка лучшей модели
#     logger.info("Loading best model...")
#     best_checkpoint_path = os.path.join(
#         config.training.checkpoint_dir,
#         f"{config.name}_best.pt"
#     )
    
#     if os.path.exists(best_checkpoint_path):
#         checkpoint = torch.load(best_checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         logger.info(f"Best model loaded from: {best_checkpoint_path}")
    
#     # Тестирование
#     logger.info("Testing on test set...")
#     model.eval()
    
#     test_metrics = trainer.validate_epoch(-1)  # -1 для обозначения теста
    
#     logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
#     logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
#     # Сохранение финальных метрик
#     final_metrics = {
#         'best_val_accuracy': max(h['val_accuracy'] for h in history),
#         'test_loss': test_metrics['loss'],
#         'test_accuracy': test_metrics['accuracy']
#     }
    
#     metrics_path = os.path.join(
#         config.training.log_dir,
#         f"{config.name}_final_metrics.json"
#     )
#     with open(metrics_path, 'w') as f:
#         json.dump(final_metrics, f, indent=2)
#     logger.info(f"Final metrics saved to: {metrics_path}")
    
#     logger.info("="*80)
#     logger.info("Training completed successfully!")
#     logger.info(f"Best checkpoint: {best_checkpoint_path}")
#     logger.info("="*80)


# if __name__ == "__main__":
#     main()