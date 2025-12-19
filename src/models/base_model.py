import torch
import logging
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Any
from jaxtyping import Int, Bool
from abc import ABC, abstractmethod

class BasicNeuralClassifier(nn.Module, ABC):
    """Базовый классификатор"""
    
    def __init__(self, vocab_size: int, labels_number: int, device: str = "cpu", 
                 criterion: Optional[nn.Module] = None, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.labels_number = labels_number
        self.device = device
        
        if criterion is None:
            criterion = nn.NLLLoss(reduction="mean")
        self.criterion = criterion
        self.build_network(vocab_size, labels_number, **kwargs)

    @abstractmethod
    def build_network(self, vocab_size: int, labels_number: int, **kwargs):
        pass

    @abstractmethod
    def forward(self, input_ids: Int[Tensor, "batch seq"], 
                mask: Optional[Bool[Tensor, "batch seq"]] = None, **kwargs) -> Dict[str, Any]:
        pass

    @torch.no_grad()
    def predict(self, input_ids: Int[Tensor, "batch seq"], mask: Optional[Bool[Tensor, "batch seq"]] = None):
        self.eval()
        input_ids = input_ids.to(self.device)
        mask = self._prepare_mask(input_ids, mask)
        outputs = self(input_ids, mask=mask)
        preds = torch.argmax(outputs["log_probs"], dim=-1)  # (B, L)
        return preds, mask

    def _prepare_mask(self,input_ids: Int[Tensor, "batch seq"], 
                      mask: Optional[Bool[Tensor, "batch seq"]] = None) -> Bool[Tensor, "batch seq"]:
        if mask is None:
            mask = (input_ids != 0)
        return mask.to(self.device)