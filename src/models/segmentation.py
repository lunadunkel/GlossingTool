from typing import cast
import torch
import torch.nn as nn
from src.models.base_model import BasicNeuralClassifier
from src.training.utils.initializing import register_model

@register_model("segmentation")
class MorphSegmentationCNN(BasicNeuralClassifier):
    def __init__(self, vocab_size, labels_number, device="cpu", criterion=None, **kwargs):
        super().__init__(vocab_size, labels_number, device, criterion, **kwargs)

    def build_network(self, vocab_size, labels_number, embeddings_dim=32, n_layers=1, 
                      window=5, hidden_dim=128, dropout=0.0, use_batch_norm=False,
                      use_attention=False, use_lstm=False, **kwargs):

        self.n_layers = n_layers

        self.use_attention = use_attention
        self.use_lstm = use_lstm

        match hidden_dim:
            case int():
                hidden_dim = (hidden_dim,)

        self.hidden_dim = [hidden_dim] * self.n_layers

        match window:
            case int():
                window = (window,)

        self.window = [window] * self.n_layers

        self.use_batch_norm = use_batch_norm
        self.embedding = nn.Embedding(vocab_size, embeddings_dim, padding_idx=0)

        self.convolutions = nn.ModuleList()
        output_dim = embeddings_dim
        for i in range(self.n_layers):
            input_dim = output_dim if i > 0 else embeddings_dim
            convolutions = nn.ModuleList()
            output_dim = 0
            for n_out, width in zip(self.hidden_dim[i], self.window[i]):
                convolution = nn.Conv1d(input_dim, n_out, width,
                                        padding=(width-1)//2, stride=1)
                convolutions.append(convolution)
                output_dim += n_out
            layer = {
                "convolutions": convolutions,
                "activation": torch.nn.ReLU(),
                "dropout": nn.Dropout(p=dropout)
            }
            if self.use_batch_norm:
                layer["batch_norm"] = nn.BatchNorm1d(output_dim)
            self.convolutions.append(nn.ModuleDict(layer))

        final_dim = output_dim
        if self.use_lstm:
            final_dim = 256 * 2  # hidden_size * directions

        self.lstm = nn.LSTM(output_dim, hidden_size=256,
                            num_layers=1, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(embed_dim=final_dim, num_heads=4)

        self.dense = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Linear(128, labels_number)
        )

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, mask=None, **kwargs):
        if mask is None:
            mask = (input_ids != 0)
        mask = mask.bool().to(input_ids.device)
        input_ids = input_ids.to(self.device)
        embeddings = self.embedding(input_ids) # B * L * d_emb
            
        conv_inputs = embeddings.permute([0, 2, 1]) # B * d_emb * L
        for layer in self.convolutions:
            layer = cast(nn.ModuleDict, layer)
            convolutions = cast(nn.ModuleList, layer["convolutions"])
            conv_outputs_list = []
            for convolution in convolutions:
                conv_outputs_list.append(convolution(conv_inputs))
            conv_outputs = torch.cat(conv_outputs_list, dim=1)
            if self.use_batch_norm:
                conv_outputs = layer["batch_norm"](conv_outputs)
            conv_outputs = layer["activation"](conv_outputs)
            conv_outputs = layer["dropout"](conv_outputs)
            conv_inputs = conv_outputs
        conv_outputs = conv_outputs.permute([0, 2, 1]) # type: ignore

        if self.use_lstm:
            conv_outputs, _ = self.lstm(conv_outputs)

        if self.use_attention:
            # conv_outputs: (B, L, D)
            conv_outputs = conv_outputs.permute(1, 0, 2)  # (L, B, D)
            conv_outputs, _ = self.attention(conv_outputs, conv_outputs, conv_outputs,
                                            key_padding_mask=~mask)
            conv_outputs = conv_outputs.permute(1, 0, 2) 
        # финальный слой
        logits = self.dense(conv_outputs) # B * L * labels_number
        log_probs = self.log_softmax(logits)
        _, labels = torch.max(log_probs, dim=-1)
        return {"log_probs": log_probs, "labels": labels}