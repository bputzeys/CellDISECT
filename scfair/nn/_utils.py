from typing import Any, Dict, List
import torch
import random
from scvi.nn import one_hot
import networkx as nx


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


def one_hot_cat(n_cat_list: List[int], cat_covs: torch.Tensor):
    cat_list = list()
    if cat_covs is not None:
        cat_list = list(torch.split(cat_covs, 1, dim=1))
    one_hot_cat_list = []
    if len(n_cat_list) > len(cat_list):
        raise ValueError("nb. categorical args provided doesn't match init. params.")
    for n_cat, cat in zip(n_cat_list, cat_list):
        if n_cat and cat is None:
            raise ValueError("cat not provided while n_cat != 0 in init. params.")
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                onehot_cat = one_hot(cat, n_cat)
            else:
                onehot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [onehot_cat]
    u_cat = torch.cat(*one_hot_cat_list) if len(one_hot_cat_list) > 1 else one_hot_cat_list[0]
    return u_cat


def get_paired_indices(cont_covs: torch.Tensor, cat_covs: torch.Tensor, dim_indices: int):
    # TODO: needs 2 changes:
        # correct dims
        # some cont_covs (pc_i) might be grouped to 1 zs
    n = int(cat_covs.size(dim=dim_indices))
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    # print(1)
    edge_weights = compute_covs_distances(cat_covs, cont_covs, n)
    # print(2)
    for (i, j) in edge_weights.keys():
        graph.add_edge(i, j, weight=edge_weights[(i, j)])
    # print(8)
    matching_edges = nx.min_weight_matching(graph)
    # print(9)
    half1 = [e[0] for e in matching_edges]
    half2 = [node for node in graph.nodes if node not in half1]
    return torch.tensor(half1), torch.tensor(half2)


def compute_covs_distances(cat_covs: torch.Tensor, cont_covs: torch.Tensor, n):
    covs = torch.cat((cat_covs, cont_covs), dim=-1)
    covs_count = covs.size(dim=-1)
    cat_size = cat_covs.size(dim=-1)
    dist_per_cov = {c: {(i, j): covs_distance(covs[i][c], covs[j][c], c, cat_size)
                        for i in range(n) for j in range(n)}
                    for c in range(covs_count)}
    # print('start')
    for c in range(cat_size, covs_count):
        # print(c)
        max_val = max(dist_per_cov[c].values())
        dist_per_cov[c] = {(i, j): dist_per_cov[c][(i, j)] / max_val
                           for i in range(n) for j in range(n)}
    # print('end')
    dist = {(i, j): sum(dist_per_cov[c][(i, j)] for c in range(covs_count))
            for i in range(n) for j in range(i + 1, n)}
    return dist


def covs_distance(c1, c2, cov_index, cat_size):
    if cov_index >= cat_size:
        return abs(c1 - c2)
    else:
        return 0 if c1 == c2 else 1
