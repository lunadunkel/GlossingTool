from typing import Sequence
from src.core.base_classes import SystemPipeline
from src.core.data.datasets import SegmEntry
from src.core.data.preprocessing import GlossEntry
from src.encoders.segmentation_encoder import MorphBIOEncoder
from src.training.utils.initializing import register_pipeline


@register_pipeline("segmentation")
class SegmentationPipeline(SystemPipeline):
    """Пайплайн для морфологической сегментации: GlossEntry –> SegmEntry.
    i.e из чисто прочитанных данных мы делаем данные для обучение сегментации"""
    def __init__(self):
        self.encoder_cls = MorphBIOEncoder

    def run(self, inputs: Sequence[GlossEntry]) -> Sequence[SegmEntry]:
        encoder = self.encoder_cls()
        for entry in inputs:
            try:
                encoder.add_data(entry)
            except Exception as e:
                raise RuntimeError(f"Ошибка при обработке записи {entry.id}: {e}")
        return encoder.return_data()