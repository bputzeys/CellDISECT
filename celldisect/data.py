from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
import torch 
from typing import Union

def parse_use_gpu_arg(
    use_gpu: Optional[Union[str, int, bool]] = None, return_device=True
):
    """
    Parses the use_gpu arg in codebase to be compaitible with PytorchLightning's gpus arg.
    If return_device is True, will also return the device.

    Parameters:
    -----------

    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str), or use CPU (if False).
    return_device
        If True, will return the torch.device of use_gpu.
    """

    gpu_available = torch.cuda.is_available()
    if (use_gpu is None and not gpu_available) or (use_gpu is False):
        gpus = 0
        device = torch.device("cpu")
    elif (use_gpu is None and gpu_available) or (use_gpu is True):
        current = torch.cuda.current_device()
        device = torch.device(current)
        gpus = [current]
    elif isinstance(use_gpu, int) or isinstance(use_gpu, str):
        device = torch.device(use_gpu)
        gpus = [use_gpu]

    if return_device:
        return gpus, device
    else:
        return gpus


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        accelerator, _, self.device = parse_use_gpu_arg(
            self.use_gpu, return_device=True
        )
        self.pin_memory = (
            True
            if (settings.dl_pin_memory_gpu_training and accelerator == "gpu")
            else False
        )

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            return AnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
