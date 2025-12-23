import json
from sklearn.metrics import accuracy_score
from torch import Tensor
from jaxtyping import Int
from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class TrainingHistory:
    """Класс для хранения истории обучения"""
    
    epochs: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    val_word_accuracies: List[float] = field(default_factory=list)
    
    def __len__(self) -> int:
        """Количество записанных эпох."""
        return len(self.epochs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Доступ к данным конкретной эпохи по индексу."""
        if idx < 0 or idx >= len(self.epochs):
            raise IndexError(f"Индекс {idx} вне диапазона: всего {len(self.epochs)} эпох")
        
        return {
            'epoch': self.epochs[idx],
            'train_loss': self.train_losses[idx],
            'val_loss': self.val_losses[idx],
            'val_accuracy': self.val_accuracies[idx],
            'val_word_accuracy': self.val_word_accuracies[idx]
        }
    
    def __iadd__(self, epoch_data: Dict[str, Any]) -> 'TrainingHistory':
        """Добавление данных новой эпохи через +=."""
        self.epochs.append(epoch_data['epoch'])
        self.train_losses.append(epoch_data['train_loss'])
        self.val_losses.append(epoch_data['val_loss'])
        self.val_accuracies.append(epoch_data['val_accuracy'])
        self.val_word_accuracies.append(epoch_data['val_word_accuracy'])
        return self
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Конвертация в список словарей (для сохранения в JSON)."""
        return [self[i] for i in range(len(self))]
    
    def save_json(self, filepath: str) -> None:
        """Сохранение истории в JSON файл."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict_list(), f, indent=4, ensure_ascii=False)
    
    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> 'TrainingHistory':
        """Загрузка истории из списка словарей."""
        history = cls()
        for epoch_data in data:
            history += epoch_data
        return history
    
    @classmethod
    def load_json(cls, filepath: str) -> 'TrainingHistory':
        """Загрузка истории из JSON файла."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict_list(data)
    
    def __str__(self) -> str:
        """Краткая информация об истории."""
        if len(self) == 0:
            return "TrainingHistory(empty)"
        
        last_epoch = self[-1]
        return (f"TrainingHistory({len(self)} epochs, "
                f"Best word_acc: {max(self.val_word_accuracies):.4f} at epoch {self.epochs[self.val_word_accuracies.index(max(self.val_word_accuracies))]}, "
                f"Last: {last_epoch['val_word_accuracy']:.4f})")
    
    def __repr__(self) -> str:
        return f"TrainingHistory(epochs={len(self)})"


def calculate_accuracy(labels: List[List[int]], predictions: List[List[int]], 
                             masks: List[List[bool]]):
    true_labels, true_preds = [], []
    for lbl_seq, pred_seq, mask_seq in zip(labels, predictions, masks):
        for lbl, pred, m in zip(lbl_seq, pred_seq, mask_seq):
            if m:
                true_labels.append(lbl)
                true_preds.append(pred)
    return accuracy_score(true_labels, true_preds) if true_labels else 0.0


def calculate_word_level_accuracy(labels: List[List[int]], 
    predictions: List[List[int]], o_tag: int = 0):
    """
    Точность на уровне слов (сегментов).
    Слово = непрерывная последовательность.
    """
    correct_words = 0
    total_words = 0

    for lbl_seq, pred_seq in zip(labels, predictions):
        i = 0
        while i < len(lbl_seq):
            if lbl_seq[i] == o_tag:
                i += 1
                continue

            start = i
            while i < len(lbl_seq) and lbl_seq[i] != o_tag:
                i += 1
            end = i

            true_word = lbl_seq[start:end]
            pred_word = pred_seq[start:end]
            
            if true_word == pred_word:
                correct_words += 1
            total_words += 1

    return correct_words / total_words if total_words > 0 else 0.0

