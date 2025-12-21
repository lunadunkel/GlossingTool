from abc import ABC, abstractmethod
from typing import Optional, Any, Sequence
from torch.utils.data import DataLoader
import numpy as np
import torch
from typing import Any, Dict, List, Protocol
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, fields
from jaxtyping import Int

from src.core.data.datasets import BasicEntry

class DataModule(ABC):
    """Базовый класс для модулей данных: cтандартный интерфейс для даталоадеров."""
    def __init__(self, train_entries: Sequence[Any], val_entries: Sequence[Any], 
                 test_entries: Sequence[Any], batch_size: int = 16, **kwargs):
        self.batch_size = batch_size
        self.train_entries = train_entries
        self.val_entries = val_entries
        self.test_entries = test_entries

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для обучающей выборки."""
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для валидационной выборки."""
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Возвращает DataLoader для тестовой выборки."""
        pass

class DataEncoder(ABC):
    """Базовый класс для энкодеров данных."""
    def __init__(self):
        self.data: List[Any] = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def clear(self) -> None:
        self.data.clear()

    def return_data(self) -> Sequence[Any]:
        """Возвращает все накопленные данные."""
        return self.data

    @abstractmethod
    def add_data(self, *args, **kwargs) -> None:
        pass

class DataclassLike(Protocol):
    """Класс для типизации: collate_fn принимает только список экземпляров dataclass"""
    __dataclass_fields__: Dict[str, Any]
    device: str
    input_ids: Int[torch.Tensor, "seq_len"]

class EntryDataset(Dataset):
    """Базовый класс для данных любого типа"""
    def __init__(self, entries: Sequence[BasicEntry]):
        self.entries = entries

    def __getitem__(self, idx: int):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def collate_fn(samples: Sequence[DataclassLike]):
        first_elem = samples[0]
        keys = first_elem.__dataclass_fields__.keys()
        device = first_elem.device
        dtype = first_elem.input_ids.dtype
        lengths = [elem.input_ids.shape[0] for elem in samples]
        L = max(lengths)
        answer = dict()
        for key in keys:
            if key not in ['id', 'device']:
                answer[key] = torch.stack([
                    torch.cat([
                        getattr(elem, key),
                        torch.zeros(size=(L-len(getattr(elem, key)),), dtype=dtype).to(device)
                    ]) for elem in samples
                ])

        answer['id'] = np.array([getattr(elem, 'id') for elem in samples])
        return answer

class SystemPipeline(ABC):
    """Абстрактный базовый класс для всех пайплайнов обработки данных перед обучением / обработкой."""
    @abstractmethod
    def run(self, inputs: Sequence[Any]) -> Sequence[Any]:
        """Запуск пайплайна: преобразование входных данных в выходные.
        Args:
            inputs: список входных объектов (тип зависит от задачи)
        Returns:
            список выходных объектов
        """
        raise NotImplementedError