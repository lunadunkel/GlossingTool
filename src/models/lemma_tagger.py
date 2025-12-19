import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BasicNeuralClassifier
from src.training.utils.initializing import register_model

@register_model("lemma_tagger")
class LemmaAffixTagger(BasicNeuralClassifier):
    def __init__(self, vocab_size: int, labels_number: int = 3, device: str = "cpu", **kwargs):
        super().__init__(vocab_size, labels_number, device, **kwargs)

    def build_network(self, vocab_size, labels_number, emb_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.0, **kwargs):
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, labels_number)
        self.lstm_dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(reduction="mean")

    def forward(self, input_ids, mask=None, **kwargs):
        embeddings = self.embedding(input_ids)          
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.lstm_dropout(lstm_out)             

        logits = self.classifier(lstm_out)                 
        log_probs = self.log_softmax(logits)

        return {"log_probs": log_probs}