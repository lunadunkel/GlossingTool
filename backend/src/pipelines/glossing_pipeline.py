from pathlib import Path
import re
import sys
from typing import List, Sequence

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

backend_str = str(BACKEND_DIR)
if backend_str not in sys.path:
    sys.path.insert(0, backend_str)


from src.core.data.project_exceptions import LengthMismatch
from src.core.base_classes import SystemPipeline
from src.core.data.preprocessing import GlossEntry, InterGloss


class GlossingPipeline(SystemPipeline):
    def __init__(self):
        punctuation = '!"#$%&\'()*+,/;<=>?@[\\]^_`{|}~'
        self.punct_pattern = re.compile(r'(?:[«' + re.escape(punctuation) + '»]+|[.]$)')
        self.bracketed = re.compile(r'(?:\([^()]+\))*')
        self.speech = re.compile(r"\[[А-Я]+\:\]")
    
    def run(self, inputs: Sequence[GlossEntry]) -> Sequence[InterGloss]:
        data = []
        for entry in inputs:
            orig_segm = entry.segmented
            gloss = entry.glossed
            
            segm = re.sub(self.speech, '', orig_segm)
            segm = re.sub(self.bracketed, '', segm)
            segm = re.sub(self.punct_pattern, '', segm)

            gloss = re.sub(self.speech, '', gloss)
            gloss = re.sub(self.bracketed, '', gloss)
            gloss = re.sub(self.punct_pattern, '', gloss)

            pattern = re.compile(r'^[а-яА-ЯёЁ.]+$')
            if len(segm.split()) != len(gloss.split()):
                raise LengthMismatch(entry.text_name, entry.line_num, segm.split(), gloss.split())
            indices = []
            for segm_word, gloss_word in zip(segm.split(), gloss.split()):
                word_indices = []
                for i, gloss_p in enumerate(gloss_word.split('-')):
                    if pattern.fullmatch(gloss_p):
                        word_indices.append(i)
                indices.append(word_indices)
            if len(indices) != len(segm.split()):
                raise LengthMismatch(entry.text_name, entry.line_num, segm.split(), gloss.split())

            gloss_entry = InterGloss(id=entry.id,
                               orig_segm=orig_segm,
                               segmentation=segm,
                               glossed=gloss,
                               translation=entry.translation,
                               lemma_indices=indices)
            data.append(gloss_entry)
        return data