from dataclasses import asdict
from typing import Any, Dict, List, Type, Union
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

from src.core.base_classes import BaseInference
from src.core.config import InferenceConfig
from src.core.initializing import register_inference
from src.models.base_model import BasicNeuralClassifier
from src.tokenization.char_tokenization import CharTokenizer
from src.tokenization.utils.load_best_checkpoint import load_best_checkpoint
from vocabularies.id2char import symbols
from torch.nn.utils.rnn import pad_sequence


class InferenceDataset(Dataset):
    def __init__(self, entries: List[Any]):
        self.entries = entries

    def __getitem__(self, idx: int):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)

@register_inference("segmentation")
class InferenceSegmentation(BaseInference):
    def __init__(self, config: InferenceConfig, model: Type[BasicNeuralClassifier], logger, device='cpu'):
        curr_model = model(**asdict(config.model))
        self.optimizer = torch.optim.AdamW(curr_model.parameters(), lr=float(config.training.learning_rate))
        self.tokenizer = CharTokenizer(symbols)

        if device == 'cuda' and torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'


        self.model, self.optimizer = load_best_checkpoint(curr_model, self.optimizer, logger=logger, 
                                                          checkpoint=f'{config.training.checkpoint_dir}/{config.task}_best.pt',
                                                          device=self.device)
        self.model.eval()  # type: ignore

    def _predict_on_batch(self, dataloader: DataLoader):
        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                mask = self.model._prepare_mask(input_ids=input_ids)
                outputs = self.model(input_ids, mask=mask)
                preds = torch.argmax(outputs['log_probs'], dim=-1).cpu().tolist()
                mask_list = mask.cpu().tolist()

                filtered_preds = [
                    [p for p, m in zip(pred_seq, mask_seq) if m]
                    for pred_seq, mask_seq in zip(preds, mask_list)
                ]

                all_predictions.extend(filtered_preds) 
        return all_predictions
                
    def _collate_fn(self, batch: List[torch.Tensor]):
        input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
        return {'input_ids': input_ids}
    
    def _tags_to_string(self, original: List[str], labels: List[List[int]]):
        final_data = []
        for ex, preds in zip(original, labels):
            final_string = ''
            prev_label = None
            for char, l in zip(ex, preds):
                # print(char, l)
                match l:
                    case 1:
                        if prev_label is None or prev_label == 0:
                            final_string += char
                            prev_label = l
                        else:
                            final_string += f'-{char}'
                    case 2:
                        final_string += char
                        prev_label = l
                        if prev_label is None or prev_label == 0:
                            prev_label = 1
                    case 0:
                        final_string += '\t'
                        prev_label = l
            final_data.append(final_string)
        return final_data

    def predict(self, data: List[str] | str):
        batch = []
        all_data = []
        if isinstance(data, list):
            sents = self.tokenizer.tokenize(data)
            all_data.extend(sents)
            for sent in sents:
                input_ids = self.tokenizer.get_tokens_ids([sent])
                batch.append(torch.tensor(input_ids, dtype=torch.int64).to(self.device))
        elif isinstance(data, str):
            sent = self.tokenizer.tokenize(data.split())
            all_data.append(sent)
            input_ids = self.tokenizer.get_tokens_ids(sent)
            batch.append(torch.tensor(input_ids, dtype=torch.int64).to(self.device))
        else:
            formatted_args = ', '.join([x.__name__ for x in get_args(InferenceSegmentation.predict.__annotations__['data'])])
            raise ValueError(f'Функция predict принимает только значения типа {formatted_args}, а получила {type(data)}')
        
        dataset = InferenceDataset(batch)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=self._collate_fn)
        labels = self._predict_on_batch(dataloader)
        data = self._tags_to_string(all_data, labels)
        return data