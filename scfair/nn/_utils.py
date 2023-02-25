import torch
import random
from scvi.nn import one_hot


def one_hot_cat(n_cat_list, cat_covs, batch_index):  # n_cat_list = self.cat_list
    cat_list = list()
    if cat_covs is not None:
        cat_list = torch.split(cat_covs, 1, dim=1)
    batch_list = torch.split(batch_index, 1, dim=1)
    cat_list = list(batch_list) + list(cat_list)
    one_hot_cat_list = []
    if len(n_cat_list) > len(cat_list):
        raise ValueError("nb. categorical args provided doesn't match init. params.")
    for n_cat, cat in zip(n_cat_list, cat_list):
        if n_cat and cat is None:
            raise ValueError("cat not provided while n_cat != 0 in init. params.")
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                one_hot_cat = one_hot(cat, n_cat)
            else:
                one_hot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [one_hot_cat]
    u_cat = torch.cat(*one_hot_cat_list) if len(one_hot_cat_list) > 1 else one_hot_cat_list[0]
    return u_cat


def get_counterfactual_cat(n_cat_list, cat_covs, batch_index, onehot=False):  # n_cat_list = self.cat_list
    cat_list = list()
    if cat_covs is not None:
        cat_list = torch.split(cat_covs, 1, dim=1)
    batch_list = torch.split(batch_index, 1, dim=1)
    cat_list = list(batch_list) + list(cat_list)
    counterfactual_cat_list = []
    if len(n_cat_list) > len(cat_list):
        raise ValueError("nb. categorical args provided doesn't match init. params.")
    for n_cat, cat in zip(n_cat_list, cat_list):
        if n_cat and cat is None:
            raise ValueError("cat not provided while n_cat != 0 in init. params.")
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            rand_shift = random.randint(1, n_cat - 1)
            shifted_cat = (cat + rand_shift) % n_cat
            if shifted_cat.size(1) != n_cat and onehot:
                shifted_cat = one_hot(shifted_cat, n_cat)
            counterfactual_cat_list += [shifted_cat]
    cat_c = torch.cat(*counterfactual_cat_list) if len(counterfactual_cat_list) > 1 else counterfactual_cat_list[0]
    return cat_c

