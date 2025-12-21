from enum import Enum
import re
from typing import List, Optional, Tuple
from string import punctuation
import torch
from src.core.data.datasets import TaggerEntry
from src.core.data.preprocessing import InterGloss
from vocabularies.id2char import symbols
from src.core.base_classes import DataEncoder
from src.tokenization.char_tokenization import CharTokenizer


class LabelTagger(Enum):
    SEP = 0
    LEMMA = 1
    AFFIX = 2

class CharLemmaAffixEncoder(DataEncoder):
    def __init__(self):
        super().__init__()
        self.tokenizer = CharTokenizer(symbols)

    def _get_tokenizations(self, token_text: str, lemma_indices: Optional[List[int]]) -> Tuple[List[int], List[int]]:
        """
        Токенизация слова посимвольно и генерация меток.
        lemma_indices: индексы символов, которые относятся к лемме
        """
        chars = self.tokenizer.tokenize([token_text])[0]  # токенизация одного слова
        input_ids = self.tokenizer.get_tokens_ids([chars])

        labels = []

        if lemma_indices is not None:
            index = 0
            for char in chars:
                if char == '-':
                    index += 1
                    labels.append(LabelTagger.SEP.value)
                elif index in lemma_indices:
                    labels.append(LabelTagger.LEMMA.value)
                else:
                    labels.append(LabelTagger.AFFIX.value)

        return input_ids, labels

    def add_data(self, entry: InterGloss, device: str = 'cpu') -> None:
        """
        Добавление одного примера InterGloss в датасет
        """
        segments = entry.segmentation.split()
        lemma_indices = entry.lemma_indices
        if lemma_indices is None:
            lemma_indices = [[] * len(segments)]
        for segm, lemma_bool in zip(entry.segmentation.split(), lemma_indices):
            input_ids, labels = self._get_tokenizations(segm, lemma_bool)
            mask = torch.tensor([input_ids != 0])

            device_to_use = device
            if device_to_use == 'cuda' and not torch.cuda.is_available():
                device_to_use = 'cpu'

            tagger_entry = TaggerEntry(
                id=entry.id,
                input_ids=torch.tensor(input_ids, dtype=torch.int64, device=device_to_use),
                labels=torch.tensor(labels, dtype=torch.int64, device=device_to_use),
                mask=mask, device=device_to_use)
            
            self.data.append(tagger_entry)

    def return_data(self) -> List[TaggerEntry]:
        return self.data