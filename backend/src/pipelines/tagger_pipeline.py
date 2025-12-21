from typing import List, Sequence
from src.core.base_classes import SystemPipeline

from src.core.data.datasets import TaggerEntry
from src.core.data.preprocessing import GlossEntry, InterGloss
from src.encoders.tagger_encoder import CharLemmaAffixEncoder
from src.pipelines.glossing_pipeline import GlossingPipeline
from src.core.initializing import register_pipeline

@register_pipeline("lemma_tagger")
class TaggerPipeline(SystemPipeline):
    """Пайплайн для (псевдо)бинарной классификации: InterGloss –> TaggerEntry."""
    def __init__(self):
        self.encoder_cls = CharLemmaAffixEncoder

    def run(self, inputs: Sequence[GlossEntry]) -> List[TaggerEntry]:
        InterGloss = GlossingPipeline()
        inter_inputs = InterGloss.run(inputs)
        encoder = self.encoder_cls()
        for entry in inter_inputs:
            try:
                encoder.add_data(entry)
            except Exception as e:
                raise RuntimeError(f"Ошибка при обработке записи {entry.id}: {e}") from e
        return encoder.return_data()