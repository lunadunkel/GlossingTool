from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple

class TokenType(Enum):
    """Тип токена"""
    WORD = "word"
    SPEAKER = "speaker"
    BRACKETED = "bracketed"

@dataclass
class Token:
    text: str
    type: TokenType
    mask_for_bio: bool
    
    def __repr__(self):
        return f"Token(text='{self.text}', type={self.type}, mask_for_bio={self.mask_for_bio})"

@dataclass
class InterGloss:
    id: int
    orig_segm: str
    segmentation: str
    translation: Optional[str]
    glossed: Optional[str]
    lemma_indices: Optional[Iterable[List[int]]]

    def __repr__(self):
        return f"InterGloss(segmentation='{self.segmentation}', glossed={self.glossed})"

class GlossingPreprocessor:
    bracketed_re = re.compile(r"^\([^()]\)$")
    speaker_re = re.compile(r"^\[.*\]$")

    def _tokenize(self, line: str) -> List[str]:
        """Разбивает строку на токены"""
        return line.strip().split()
    
    def _find_bracketed_tokens(self, tokens: List[str]) -> list:
        parenthesis = False
        indexes = []
        for token in tokens:
            token = re.sub('[,.]+', '', token)
            if token.startswith('(') and token.endswith(')'):
                indexes.append(True)
                parenthesis = False
            elif token.startswith('(') and not token.endswith(')'):
                parenthesis = True
                indexes.append(True)
            elif not token.startswith('(') and token.endswith(')'):
                parenthesis = False
                indexes.append(True)
            else:
                if parenthesis:
                    indexes.append(True)
                else:
                    indexes.append(False)
        return indexes

    def _classify_token(self, token: str, bracketed=False) -> Token:
        """Классификация токена"""
        if bracketed:
            return Token(text=token, type=TokenType.BRACKETED, mask_for_bio=True)
        if self.bracketed_re.match(token):
            return Token(text=token, type=TokenType.BRACKETED, mask_for_bio=True)
        if self.speaker_re.match(token):
            return Token(text=token, type=TokenType.SPEAKER, mask_for_bio=True)
        return Token(text=token, type=TokenType.WORD, mask_for_bio=False)

    def process_line(self, line: str) -> List[Token]:
        """Обработка строки глоссирования или сегментации"""
        tokens = self._tokenize(line)
        bracketed_tokens = self._find_bracketed_tokens(tokens)
        return [self._classify_token(tok, bracket) for tok, bracket in zip(tokens, bracketed_tokens)]

@dataclass
class GlossEntry:
    id: int
    segmented: str
    glossed: str
    translation: Optional[str]
    metadata: Optional[str]
    text_name: str
    line_num: str
    orig_tokens: List[str] = field(init=False)
    segm_tokens: List[Token] = field(init=False)
    gloss_tokens: List[Token] = field(init=False)
    device: str = 'cpu'

    def __post_init__(self):
        prep = GlossingPreprocessor()
        self.orig_tokens = re.sub(r'[,\.-]+', '', self.segmented).split('\t')
        self.segm_tokens = prep.process_line(self.segmented)
        self.gloss_tokens = prep.process_line(self.glossed)

class ContentReader:
    def __init__(self):
        self.data: List[GlossEntry] = []
        self.text_name: str = 'n_r.txt'
        self._update_dictionary()
    
    def _update_id(self) -> int:
        return len(self.data)

    def _update_dictionary(self):
        self.current = {"segmented": "",
                        "glossed": "",
                        "metadata": "",
                        "translation": "",
                        "text_name": "",
                        "line_num": ""}

    def _push_current(self):
        if self.current and self.current.get('segmented'):
            entry = GlossEntry(id=self._update_id(),
                segmented=self.current.get("segmented") or "",
                glossed=self.current.get("glossed") or "",
                translation=self.current.get("translation", ""),
                metadata=self.current.get("metadata", ""),
                text_name=self.text_name or "",
                line_num=self.current.get("line_num") or ""
            )
            self.data.append(entry)
            self._update_dictionary()

    def _add_line(self, key: str, value: str):
        value = re.sub('\t+', '\t', value)
        if key in self.current and self.current[key] != '':
            self.current[key] += f"\t{value}"
        else:
            self.current[key] = value

    def _check_content_type(self, line: str) -> Tuple[str, str]:
        if segmentation := re.match(r"\d+(?:_\d+)*>\s*(.*)$", line):
            return 'segmented', segmentation.group(1)
        if glossing := re.match(r"\d+(?:_\d+)*<\s*(.*)$", line):
            return 'glossed', glossing.group(1)

        if line.startswith("#"):
            meta = line.lstrip("# \t")
            return 'metadata', meta
        
        if translation_line := re.match(r"(\d+(?:_\d+)*)=(.*)$", line):
            num, translation = translation_line.groups()
            self._add_line('line_num', num)
            return 'translation', translation
        if re.match(r'\d+@', line):
            return 'corrections', ''
        return 'unknown', ''
        
    def _read_content_from_path(self, path: Path):
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                line = line.lstrip('\ufeff')
                if not line:
                    if self.current.get('segmented') != '':
                        self._push_current()
                        self._update_dictionary()
                    continue
                content_type, data = self._check_content_type(line)
                match content_type:
                    case 'unknown':
                        continue
                    case 'corrections':
                        continue
                    case _:
                        self._add_line(content_type, data)
        self._push_current()
    
    def read_content(self, data: list[str] | Path) -> List[GlossEntry]:
        """Функция чтения входящих на глоссирование данных
        Args:
            data (list | str): Список локальных путей к файлу или текст в формате string
        Returns:
            List[GlossEntry]: Список объектов GlossEntry"""
        if isinstance(data, list):
            for num, elem in enumerate(data):
                self.text_name = os.path.basename(elem).removesuffix('.txt')
                self._read_content_from_path(Path(elem))
        elif isinstance(data, Path):
            self._read_content_from_path(data)
        return self.data

class GlossDataSource:
    """Загрузка GlossEntry из файлов, используя ContentReader.
    Args:
        base_data_dir (str): путь к файлу или файлам
        texts_names_path (optional: str): путь к названиям файлов"""
    def __init__(self, base_data_dir: str, texts_names_path: Optional[str] = None):
        self.texts_names_path = texts_names_path
        self.base_data_dir = Path(base_data_dir)

    def get_gloss_entries(self) -> List[GlossEntry]:
        """
        Получить предобработанный список элементов dataclass GlossEntry c записями:
        - id: int
        - segmented: str
        - glossed: str
        - translation: Optional[str]
        - metadata: Optional[str]
        - text_name: str
        - line_num: str
        - orig_tokens: List[str] = field(init=False)
        - segm_tokens: List[Token] = field(init=False)
        - gloss_tokens: List[Token] = field(init=False)
        """
        if self.texts_names_path is not None:
            with open(self.texts_names_path, 'r', encoding='utf-8') as f:
                texts_names = [line.strip() for line in f if line.strip()]

            data = []
            for name in texts_names:
                text_dir = self.base_data_dir / name
                if not text_dir.is_dir():
                    continue

                files = {f.name for f in text_dir.iterdir() if f.is_file()}
                if f"{name}.txt" in files:
                    data.append(str(text_dir / f"{name}.txt"))
                elif "n_r.txt" in files:
                    data.append(str(text_dir / "n_r.txt"))
        else:
            data = self.base_data_dir

        reader = ContentReader()
        return reader.read_content(data)