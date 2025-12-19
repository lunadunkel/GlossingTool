from dataclasses import dataclass
from jaxtyping import Int, Bool
from torch import Tensor

@dataclass
class BasicEntry:
    id: int
    input_ids: Int[Tensor, "seq_len"]
    labels: Int[Tensor, "seq_len"]
    device: str

class SegmEntry(BasicEntry):
    id: int
    input_ids: Int[Tensor, "seq_len"]
    labels: Int[Tensor, "seq_len"]
    device: str = 'cpu'
    mask: Bool[Tensor, "seq_len"]

class TaggerEntry(BasicEntry):
    id: int
    input_ids: Int[Tensor, "seq_len"]
    labels: Int[Tensor, "seq_len"]
    device: str = 'cpu'