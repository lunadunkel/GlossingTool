from dataclasses import dataclass
from enum import Enum
import re
from string import punctuation
from typing import List, Literal, Sequence, Tuple
import warnings
from torch import Tensor
from jaxtyping import Int, Bool
import torch
from src.core.data.datasets import SegmEntry
from src.core.data.preprocessing import GlossEntry, Token
from src.tokenization.char_tokenization import CharTokenizer
from vocabularies.id2char import symbols
from src.core.base_classes import DataEncoder

class BioTag(Enum):
    B = 1
    I = 2
    O = 0

class MorphBIOEncoder(DataEncoder):
    @staticmethod
    def get_tokenizations(tokens: List[Token]) -> Tuple[List[int], List[List[str]]]: 
        tokenizer = CharTokenizer(symbols)
        char_tokens = tokenizer.tokenize(tokens)
        char_idxs = tokenizer.get_tokens_ids(char_tokens)
        return char_idxs, char_tokens

    @staticmethod
    def make_bio(segmentation: List[Token], 
                 char_tokenization: List[List[str]]) -> List[Literal[0] | Literal[1] | Literal[2]]:
        """получение BIO-разметки в уже энкоудинг виде
        Args:
            segmentation (List[Token]): список сегментации типа Token"""
        new_begining, bio_tag = True, []
        for idx, (word, symbs) in enumerate(zip(segmentation, char_tokenization)):
            if word.mask_for_bio:
                bio_tag.extend([BioTag.O.value]*(len(word.text)))
                if idx != len(segmentation)-1:
                    bio_tag.append(BioTag.O.value)
                continue
            
            char_num, prev_letter = 0, ''
            for letter in word.text:
                letter = prev_letter + letter
                if re.match('-', letter):
                    new_begining = True
                    continue
                if letter in punctuation and symbs[char_num] == '<PUNC>':
                    bio_tag.append(BioTag.O.value)
                    char_num += 1
                    continue

                if letter != symbs[char_num]:
                    prev_letter += letter
                    continue
                
                prev_letter = ''
                char_num += 1

                if new_begining:
                    bio_tag.append(BioTag.B.value)
                    new_begining = False
                else:
                    bio_tag.append(BioTag.I.value)
            if idx != len(segmentation)-1:
                new_begining = True
                bio_tag.append(BioTag.O.value)
        
        chars = sum([len(x) for x in char_tokenization])
        tabs = len(char_tokenization)-1
        segm_len = chars + tabs
        bio_len = len(bio_tag)
        if segm_len != bio_len:
            raise Exception(f'Lengths of the segmentation and BIO-tag should be the same, got {segm_len} and {bio_len}')
        else:
            return bio_tag 

    def add_data(self, entry: GlossEntry, device: str = 'cpu', inference_mode=False) -> None:
        tokens = entry.segm_tokens
        char_idxs, char_tokenizations = self.get_tokenizations(entry.segm_tokens)
        char_idxs = torch.tensor(char_idxs, dtype=torch.int64)
        if not inference_mode:
            bio_tags = torch.tensor(self.make_bio(tokens, char_tokenizations), dtype=torch.int64)
        else:
            bio_tags = torch.zeros(len(char_idxs), dtype=torch.int64)
        mask = torch.tensor(char_idxs != 0).bool()
        if device == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                warnings.warn('Device \'cuda\' is not available, switching to \'cpu\'.')
                device = 'cpu'

        segm_entry = SegmEntry(id=entry.id,
                               input_ids=char_idxs,
                               labels=torch.tensor(bio_tags),
                               mask=mask,
                               device=device)
        self.data.append(segm_entry)

    def return_data(self) -> Sequence[SegmEntry]:
        return self.data