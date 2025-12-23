from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
from pathlib import Path
import yaml


@dataclass
class TokenizerConfig:
    """Конфигурация токенизатора"""
    vocab_path: str = "vocabularies/id2char.json"
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ModelConfig:
    """Базовая конфигурация модели"""
    vocab_size: int
    labels_number: int
    device: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class SegmentationConfig(ModelConfig):
    """Конфигурация модели сегментации"""
    vocab_size: int
    emb_dim: int = 32
    hidden_dim: Union[int, list, tuple] = 256
    num_labels: int = 3  # B, I, O
    n_layers: int = 3
    dropout: float = 0.4
    window: List[int] = field(default_factory = lambda: [3, 5])
    use_lstm: bool = True
    use_attention: bool = False
    use_batch_norm: bool = True
    device: str = 'cpu'


@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    patience: int = 20
    random_seed: int = 42
    device: str = 'cpu'
    
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DataConfig:
    """Конфигурация данных"""
    texts_list: str = "backend/src/core/data/texts_names.txt"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

@dataclass
class ExperimentConfig:
    """Полная конфигурация эксперимента"""
    name: str
    task: str
    
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Загрузить конфигурацию из YAML"""
        with open(path, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        task = config_dict['task']
        model_dict = config_dict['model']
        
        if task == 'segmentation':
            model = SegmentationConfig(**model_dict)
        else:
            model = ModelConfig(**model_dict)
        
        return cls(
            name=config_dict['name'],
            task=task,
            model=model,
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            tokenizer=TokenizerConfig(**config_dict.get('tokenizer', {}))
        )
    
    def save_yaml(self, path: str):
        """Сохранить конфигурацию в YAML"""
        config_dict = {
            'name': self.name,
            'task': self.task,
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'data': asdict(self.data),
            'tokenizer': self.tokenizer.to_dict()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

@dataclass
class InferenceConfig:
    """Конфигурация для инференса"""
    name: str
    task: str
    
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig
    
    @classmethod
    def from_yaml(cls, path: str) -> 'InferenceConfig':
        """Загрузить конфигурацию из YAML"""
        with open(path, encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        task = config_dict['task']
        model_dict = config_dict['model']
        
        if task == 'segmentation':
            model = SegmentationConfig(**model_dict)
        else:
            model = ModelConfig(**model_dict)
        
        return cls(
            name=config_dict['name'],
            task=task,
            model=model,
            training=TrainingConfig(**config_dict.get('training', {})),
            tokenizer=TokenizerConfig(**config_dict.get('tokenizer', {}))
        )
