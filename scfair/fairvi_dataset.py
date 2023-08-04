import logging
from typing import TYPE_CHECKING, Dict, List, Union, Tuple

import torch
import h5py
import numpy as np
import pandas as pd
from anndata._core.sparse_dataset import SparseDataset
from scipy.sparse import issparse
from torch.utils.data import Dataset

from scvi._constants import REGISTRY_KEYS

import json

if TYPE_CHECKING:
    from scvi.data._manager import AnnDataManager

logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FairAnnTorchDataset(Dataset):
    """Extension of torch dataset to get tensors from AnnData.

    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    getitem_tensors
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor) or list of
        such keys. A list can be used to subset to certain keys in the event that more tensors than
        needed have been registered. If ``None``, defaults to all registered data.

    """

    def __init__(
        self,
        adata_manager: "AnnDataManager",
        getitem_tensors: Union[List[str], Dict[str, type]] = None,
    ):
        if adata_manager.adata is None:
            raise ValueError(
                "Please run register_fields() on your AnnDataManager object first."
            )
        self.adata_manager = adata_manager
        self.is_backed = adata_manager.adata.isbacked
        self.attributes_and_types = None
        if getitem_tensors is not None:
            data_registry = adata_manager.data_registry
            for key in (
                getitem_tensors.keys()
                if isinstance(getitem_tensors, dict)
                else getitem_tensors
            ):
                if key not in data_registry.keys():
                    raise ValueError(
                        f"{key} required for model but not registered with AnnDataManager."
                    )
        self.getitem_tensors = getitem_tensors
        self._setup_getitem()
        self._set_data_attr()

        self.idx_cf_tensor = torch.tensor([]).to(device) # idx_cf_tensor[idx] = [idx_cf with cov_column i out]
        # ith idx_cf is a random index (diff from idx if possible) which has covs (except ith cov) as idx

        self.setup_idx_cf_tensor()

    def create_idx_cf_tensor(self):
        covs = torch.tensor(self.data[REGISTRY_KEYS.CAT_COVS_KEY].values).to(device)
        # init
        self.idx_cf_tensor = torch.tensor([[i for _ in range(covs.size(1))] for i in range(covs.size(0))]).to(device)
        # fill
        for j in range(covs.size(1)):
            sub_covs = torch.cat([covs[:, :j], covs[:, j + 1:]], dim=1).to(device)
            sub_covs_to_idx = self.unique_sub_covs_to_idx(sub_covs)
            for i in range(covs.size(0)):
                sub_cov_str = str(sub_covs[i])
                idx_cf_candidates = np.array([idx for idx in sub_covs_to_idx[sub_cov_str]
                                              if covs[idx][j] != covs[i][j]])
                if len(idx_cf_candidates) > 0:
                    self.idx_cf_tensor[i][j] = np.random.choice(idx_cf_candidates)

    def unique_sub_covs_to_idx(self, sub_covs):
        rows_str = np.array([str(row) for row in sub_covs])

        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(rows_str)

        # sorts records array so all unique elements are together
        sorted_sub_covs = rows_str[idx_sort]

        # returns the unique values, the index of the first occurrence of a value
        vals, idx_start = np.unique(sorted_sub_covs, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        return dict(zip(vals, res))

    def setup_idx_cf_tensor(self):
        # self.adata_manager.cov_to_idx_dicts_path
        path = str(self.adata_manager.idx_cf_tensor_path)
        if not path.endswith('.pt'):
            path += '.pt'
        try:
            self.idx_cf_tensor = torch.load(path)
        except FileNotFoundError:
            self.create_idx_cf_tensor()
            torch.save(self.idx_cf_tensor, path)

    @property
    def registered_keys(self):
        """Returns the keys of the mappings in scvi data registry."""
        return self.adata_manager.data_registry.keys()

    def _set_data_attr(self):
        """Sets data attribute.

        Reduces number of times anndata needs to be accessed
        """
        self.data = {
            key: self.adata_manager.get_from_registry(key)
            for key, _ in self.attributes_and_types.items()
        }

    def _setup_getitem(self):
        """Sets up the __getitem__ function used by PyTorch.

        By default, getitem will return every single item registered in the scvi data registry
        and will attempt to infer the correct type. np.float32 for continuous values, otherwise np.int64.

        If you want to specify which specific tensors to return you can pass in a List of keys from
        the scvi data registry. If you want to speficy specific tensors to return as well as their
        associated types, then you can pass in a dictionary with their type.
        """
        registered_keys = self.registered_keys
        getitem_tensors = self.getitem_tensors
        if isinstance(getitem_tensors, List):
            keys = getitem_tensors
            keys_to_type = {key: np.float32 for key in keys}
        elif isinstance(getitem_tensors, Dict):
            keys = getitem_tensors.keys()
            keys_to_type = getitem_tensors
        elif getitem_tensors is None:
            keys = registered_keys
            keys_to_type = {key: np.float32 for key in keys}
        else:
            raise ValueError(
                "getitem_tensors invalid type. Expected: List[str] or Dict[str, type] or None"
            )
        for key in keys:
            if key not in registered_keys:
                raise KeyError(f"{key} not in data_registry")

        self.attributes_and_types = keys_to_type

    def __getitem__(self, idx: List[int]) -> Dict[str, Tuple[np.ndarray]]:
        """Get tensors in dictionary from anndata at idx."""
        if isinstance(idx, int):
            idx = [idx]

        data_numpy = {}

        if self.is_backed:
            # need to sort idxs for h5py datasets
            idx = np.sort(idx)

        for key, dtype in self.attributes_and_types.items():
            cur_data = self.data[key]
            get_i = lambda f: (f(cur_data, idx),
                               *[f(cur_data, self.idx_cf_tensor[idx][:, j].tolist())
                                 for j in range(self.idx_cf_tensor[idx].size(1))])

            # for backed anndata
            if isinstance(cur_data, h5py.Dataset) or isinstance(
                cur_data, SparseDataset
            ):
                f1 = lambda d, i: d[i]
                sliced_data = get_i(f1)
                if issparse(sliced_data[0]):
                    sliced_data = (s.toarray().astype(dtype) for s in sliced_data)
            elif isinstance(cur_data, np.ndarray):
                f2 = lambda d, i: d[i].astype(dtype)
                sliced_data = get_i(f2)
            elif isinstance(cur_data, pd.DataFrame):
                f3 = lambda d, i: d.iloc[i, :].to_numpy().astype(dtype)
                sliced_data = get_i(f3)
            elif issparse(cur_data):
                f4 = lambda d, i: d[i].toarray().astype(dtype)
                sliced_data = get_i(f4)
            # for minified  anndata, we need this because we can have a string
            # cur_data, which is the value of the MINIFY_TYPE_KEY in adata.uns,
            # used to record the type data minification
            # TODO: Adata manager should have a list of which fields it will load
            elif isinstance(cur_data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                continue
            else:
                raise TypeError(f"{key} is not a supported type")
            # Make a row vector if only one element is selected
            # this is because our dataloader disables automatic batching
            # Normally, this would be handled by the default collate fn
            if len(idx) == 1:
                sliced_data = (s.reshape(1, -1) for s in sliced_data)
            data_numpy[key] = sliced_data

        return data_numpy

    def get_data(self, scvi_data_key: str):
        """Get the tensor associated with a key."""
        tensors = self.__getitem__(idx=list(range(self.__len__())))
        return tensors[scvi_data_key]

    def __len__(self):
        return self.adata_manager.adata.shape[0]
