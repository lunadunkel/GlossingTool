from sklearn.metrics import accuracy_score
from torch import Tensor
from jaxtyping import Int
from typing import List

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