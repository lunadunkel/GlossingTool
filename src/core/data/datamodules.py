from typing import List, Sequence
from torch.utils.data import DataLoader
from src.core.base_classes import DataModule, EntryDataset
from src.core.data.datasets import SegmEntry, TaggerEntry
from src.training.utils.initializing import register_datamodule

@register_datamodule("segmentation")
class SegmDataModule(DataModule):
    """DataModule для SegmEntry."""
    def __init__(self, train_entries: Sequence[SegmEntry], val_entries: Sequence[SegmEntry],
                 test_entries: Sequence[SegmEntry], batch_size: int = 16):
        super().__init__(train_entries=train_entries, val_entries=val_entries,
                         test_entries=test_entries, batch_size=batch_size)
        self.train_ds = EntryDataset(train_entries)
        self.val_ds = EntryDataset(val_entries)
        self.test_ds = EntryDataset(test_entries)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size,
            shuffle=True, collate_fn=EntryDataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=EntryDataset.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=EntryDataset.collate_fn)

@register_datamodule("lemma_tagger")
class TaggerDataModule(DataModule):
    """DataModule для TaggerEntry."""
    def __init__(self, train_entries: Sequence[TaggerEntry], val_entries: Sequence[TaggerEntry],
                 test_entries: Sequence[TaggerEntry], batch_size: int = 16):
        super().__init__(train_entries=train_entries, val_entries=val_entries,
                         test_entries=test_entries, batch_size=batch_size)
        self.train_ds = EntryDataset(train_entries)
        self.val_ds = EntryDataset(val_entries)
        self.test_ds = EntryDataset(test_entries)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size,
            shuffle=True, collate_fn=EntryDataset.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=EntryDataset.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size,
            shuffle=False, collate_fn=EntryDataset.collate_fn)