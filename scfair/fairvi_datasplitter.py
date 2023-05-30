from typing import Dict, List, Optional, Union

import numpy as np

from scvi.data import AnnDataManager
from scvi.dataloaders._concat_dataloader import ConcatDataLoader
from scvi.dataloaders._data_splitting import DataSplitter


class FairVIDataSplitter(DataSplitter):

    data_loader_cls = ConcatDataLoader

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_class = ConcatDataLoader

    def train_dataloader(self):
        """Create the train data loader."""
        return self.data_loader_class(
            self.adata_manager,
            indices_list=[np.asarray(self.train_idx), np.asarray(self.train_idx)],
            shuffle=True,
            drop_last=False,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create the validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices_list=[np.asarray(self.val_idx), np.asarray(self.val_idx)],
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create the test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_class(
                self.adata_manager,
                indices_list=[np.asarray(self.test_idx), np.asarray(self.test_idx)],
                shuffle=False,
                drop_last=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

