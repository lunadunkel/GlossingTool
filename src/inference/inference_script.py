import os
import sys
import argparse
import logging
from pathlib import Path
from dataclasses import asdict
import torch
import random
import numpy as np
from src.core.data.preprocessing import GlossDataSource
from src.inference.inference import Inference
from src.pipelines.tagger_pipeline import TaggerPipeline
from src.training.utils.logger import setup_logger
from src.core.config import ExperimentConfig, InferenceConfig
from src.training.utils.seed import set_seed
from src.core.data.datamodules import *
from src.training.trainer import Trainer
from src.models.pos_tagger import PosTagger
from src.models.lemma_tagger import LemmaAffixTagger
from src.pipelines.segmentation_pipeline import SegmentationPipeline
from src.models.segmentation import MorphSegmentationCNN
from src.training.utils.load_data import clone_and_load_data
from src.core.initializing import MODEL_REGISTRY, DATAMODULE_REGISTRY, PIPELINE_REGISTRY

def write_data(task, file_path, text, translation):
    os.makedirs(file_path, exist_ok=True)
    match task:
        case 'segmentation':
            with open(f'{file_path}/temp_{task}.txt', 'w') as file:
                for num, (niv, rus) in enumerate(zip(text, translation), 1):
                    file.write(f'{num}>\t{niv}\n')
                    file.write(f'{num}<\t\n')
                    file.write(f'{num}=\t{rus}\n')
                    file.write('\n')

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
        required=True,
        help="Path to data"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Device 'cpu' or 'cuda" 
    )
    args = parser.parse_args()

    config = InferenceConfig.from_yaml(args.config)
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
    logger.info(f"Начало инференса: {config.name}")
    logger.info(f"Задача: {config.task}")
    logger.info(f"Device: {device}")
    logger.info(f"="*80)
    
    set_seed(config.training.random_seed)
    logger.info(f"Random seed: {config.training.random_seed}")


    data_source = GlossDataSource(args.data_path)
    inference_data = ['\t'.join(x.segmented.split()) for x in data_source.get_gloss_entries()]
    translation_data = [x.translation for x in data_source.get_gloss_entries()]
    if not inference_data:
        raise RuntimeError("Данные отсутствуют")

    model = MODEL_REGISTRY[model_name]
    inference = Inference(config=config, model=model, logger=logger, device=device)
    
    predictions = inference.predict(inference_data)
    write_data(model_name, 'src/core/data/temp', predictions, translation_data)
    for i, p in zip(inference_data, predictions):
        print(i)
        print(p)
    logger.info(f'Результаты {config.task} записаны в \'src/core/data/temp_{config.task}.txt\'')


if __name__ == "__main__":
    main()