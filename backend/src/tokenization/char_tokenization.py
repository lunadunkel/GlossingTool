import re
from typing import List
from string import punctuation

from src.core.data.preprocessing import Token, TokenType

class CharTokenizer:
    def __init__(self, symbols: dict):
        self.id2char = symbols
        self.char2id = {char: idx for idx, char in symbols.items()}
        self.long_symbols = sorted([s for s in symbols.values() if len(s) > 1], key=len, reverse=True)

    def _map_tokens(self, symbs: List[List[str]]) -> List[int]:
        mapping = []
        for num, token in enumerate(symbs):
            for elem in token:
                if elem in self.char2id:
                    mapping.append(self.char2id[elem])
                else:
                    mapping.append(self.char2id['<UNK>'])
            if num != len(symbs)-1:
                mapping.append(self.char2id['\t'])
        return mapping

    def _check_bio_mask(self, word):
        if word.mask_for_bio:
            if word.type == TokenType.BRACKETED:
                return ['<PUNC>']*len(word.text)
            elif word.type == TokenType.SPEAKER:
                return ['<SPEAKER>']*len(word.text)
            return []
        return None

    def tokenize(self, tokens: List) -> List[List[str]]:
        """Посимвольная токенизация, максимально жадно выделяющая символы
        Args:
            tokens (List): список элементов"""
        chars = []
        for word in tokens:
            i = 0
            token_chars = []
            if isinstance(word, Token):
                word_text = word.text
                result = self._check_bio_mask(word)
                if result is not None:
                    token_chars.extend(result)
                    chars.append(token_chars)
                    continue
            else:
                word_text = word

            word_text = re.sub('-', '', word_text)
            while i < len(word_text):
                matched = False
                for symb in self.long_symbols:
                    if word_text.startswith(symb, i):
                        token_chars.append(symb)
                        i += len(symb)
                        matched = True
                        break
                if not matched:
                    if word_text[i] in punctuation:
                        token_chars.append('<PUNC>')
                        i += 1
                    else:
                        token_chars.append(word_text[i])
                        i += 1
            chars.append(token_chars)
        return chars
    
    def get_tokens_ids(self, chars: List[List[str]]):
        return self._map_tokens(chars)