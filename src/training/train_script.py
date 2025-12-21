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
from dataclasses import asdict
import torch
import random
import numpy as np
from src.pipelines.tagger_pipeline import TaggerPipeline
from src.training.utils.logger import setup_logger
from src.core.config import ExperimentConfig
from src.training.utils.seed import set_seed
from src.core.data.datamodules import *
from src.training.trainer import Trainer
from src.models.pos_tagger import PosTagger
from src.models.lemma_tagger import LemmaAffixTagger
from src.pipelines.segmentation_pipeline import SegmentationPipeline
from src.models.segmentation import MorphSegmentationCNN
from src.training.utils.load_data import clone_and_load_data
from src.core.initializing import MODEL_REGISTRY, DATAMODULE_REGISTRY, PIPELINE_REGISTRY

def main():
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device 'cpu' or 'cuda" 
    )
    args = parser.parse_args()

    # Загрузка конфигурации
    config = ExperimentConfig.from_yaml(args.config)
    model_name = config.task
    
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
    logger.info(f"Эксперимент: {config.name}")
    logger.info(f"Задача: {config.task}")
    logger.info(f"Device: {device}")
    logger.info(f"="*80)
    
    set_seed(config.training.random_seed)
    logger.info(f"Random seed: {config.training.random_seed}")
    
    logger.info("Препроцессинг данных...")
    clone = True if args.data_path is None else False
    train_data, val_data, test_data = clone_and_load_data(config=config.data,
                                                          logger=logger, clone=clone,
                                                          data_path=args.data_path)
    
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Val size: {len(val_data)}")
    logger.info(f"Test size: {len(test_data)}")
    
    
    logger.info("Создание даталоадеров...")

    if (model_name not in DATAMODULE_REGISTRY 
        or model_name not in MODEL_REGISTRY 
        or model_name not in PIPELINE_REGISTRY):
        logger.error(f"Модель не зарегестрирована")
        raise ValueError(f"Неизвестная модель: {model_name}")

    pipeline = PIPELINE_REGISTRY[model_name]
    datamodule = DATAMODULE_REGISTRY[model_name]
    model = MODEL_REGISTRY[model_name]

    curr_pipeline = pipeline()

    train_ds = curr_pipeline.run(train_data)
    val_ds = curr_pipeline.run(val_data)
    test_ds = curr_pipeline.run(test_data)

    curr_datamodule = datamodule(train_entries=train_ds,
                                 val_entries=val_ds,
                                 test_entries=test_ds,
                                 batch_size=config.training.batch_size)

    train_loader = curr_datamodule.train_dataloader()
    val_loader = curr_datamodule.val_dataloader()
    test_loader = curr_datamodule.test_dataloader()
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    logger.info("Создание модели...")
    curr_model = model(**asdict(config.model))

    logger.info("Создание класса Trainer для обучения модели")
    trainer = Trainer(model=curr_model, exp_name=model_name, train_loader=train_loader,
        val_loader=val_loader, config=config.training, logger=logger)
    
    logger.info("Обучение модели...")
    trainer.train()
    
    best_path = f'checkpoints/{model_name}_best.pt'
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        curr_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Загружена модель из чекпоинта: {best_path}")
    
    logger.info("Тестирование...")
    
    test_loss, test_acc, test_word_acc = trainer.validate(test=True)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc}\tTest Word Accuracy: {test_word_acc:.4f}")

    logger.info("="*80)
    logger.info("Обучение завершено")
    logger.info("="*80)


if __name__ == "__main__":
    main()