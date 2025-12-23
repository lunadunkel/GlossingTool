from dataclasses import dataclass
from jaxtyping import Int, Bool
from torch import Tensor

@dataclass
class BasicEntry:
    id: int
    mask: Bool[Tensor, "seq_len"]
    input_ids: Int[Tensor, "seq_len"]
    labels: Int[Tensor, "seq_len"]
    device: str = 'cpu'

@dataclass
class SegmEntry(BasicEntry):
    pass

@dataclass
class TaggerEntry(BasicEntry):
    pass