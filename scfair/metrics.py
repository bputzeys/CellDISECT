from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import entropy
import scipy.spatial as ss
from scipy.special import digamma
from math import log

from sklearn.feature_selection import *
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from fairlearn.metrics import *

# R for MI
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

infotheo = importr('infotheo')
mpmi = importr('mpmi')


def barplot_metric(metric_name: str,
                   method2metrics: Dict[str, List],
                   attr_names: List[str]):
    bar_width = 0.2
    figsize = (len(attr_names) * 3, 5)

    fig = plt.subplots(figsize=figsize)
    i = 0

    # draw bars for each method
    for method_name in method2metrics:
        metrics = method2metrics[method_name]
        bar_pos = [x + i * bar_width for x in np.arange(len(metrics))]
        i += 1
        plt.bar(bar_pos, metrics, width=bar_width,
                edgecolor='grey', label=method_name)

    # set labels
    plt.suptitle(metric_name, fontweight='bold')
    methods_count = len(method2metrics)
    metrics_count = len(list(method2metrics.values())[0])
    plt.xticks([r + bar_width * ((methods_count - 1) // 2) for r in range(metrics_count)],
               attr_names)
    plt.legend()
    plt.show()


def create_cats_idx(adata, cats):
    # create numerical index for each attr in cats

    for i in range(len(cats)):
        values = list(set(adata.obs[cats[i]]))

        val_to_idx = {v: values.index(v) for v in values}

        idx_list = [val_to_idx[v] for v in adata.obs[cats[i]]]

        adata.obs[cats[i] + '_idx'] = pd.Categorical(idx_list)

    return adata


# tutorial XGBoost: https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html

def clf_S_Z_metrics(adata, cats):
    # classifier Si | Zi

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    acc_results = []

    print(f'metrics for XGBoost classifier Si | Zi')

    for i in range(1, len(cats) + 1):

        Zi = adata.obsm[f"Z_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        train_result_i = []
        test_result_i = []

        for train, test in cv.split(Zi, Si):
            Zi_train = Zi[train]
            Zi_test = Zi[test]
            Si_train = Si[train]
            Si_test = Si[test]

            estimator = clone(clf)

            estimator.fit(Zi_train, Si_train, eval_set=[(Zi_test, Si_test)], verbose=False)

            train_score = estimator.score(Zi_train, Si_train)
            test_score = estimator.score(Zi_test, Si_test)

            train_result_i.append(train_score)
            test_result_i.append(test_score)

        train_score_i = np.mean(train_result_i)
        test_score_i = np.mean(test_result_i)

        acc_results.append((train_score_i, test_score_i))

        print(f'train acc S{i}: {train_score_i:.4f},  test acc S{i}: {test_score_i:.4f}')

    return acc_results


def clf_S_Z_not_metrics(adata, cats):
    # classifier Si | (Z - Zi)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    acc_results = []

    print(f'metrics for XGBoost classifier Si | (Z - Zi)')

    for i in range(1, len(cats) + 1):

        Z_not_i = adata.obsm[f"Z_not_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        train_result_i = []
        test_result_i = []

        for train, test in cv.split(Z_not_i, Si):
            Z_not_i_train = Z_not_i[train]
            Z_not_i_test = Z_not_i[test]
            Si_train = Si[train]
            Si_test = Si[test]

            estimator = clone(clf)

            estimator.fit(Z_not_i_train, Si_train, eval_set=[(Z_not_i_test, Si_test)], verbose=False)

            train_score = estimator.score(Z_not_i_train, Si_train)
            test_score = estimator.score(Z_not_i_test, Si_test)

            train_result_i.append(train_score)
            test_result_i.append(test_score)

        train_score_i = np.mean(train_result_i)
        test_score_i = np.mean(test_result_i)

        acc_results.append((train_score_i, test_score_i))

        print(f'train acc S{i}: {train_score_i:.4f},  test acc S{i}: {test_score_i:.4f}')

    return acc_results


def fair_clf_metrics(adata, cats, y_name):
    # fairness metrics: DP, EO, ...
    # https://fairlearn.org/v0.9/user_guide/assessment/common_fairness_metrics.html

    # binarize y
    mid = sorted(adata.obs[y_name])[len(adata.obs[y_name]) // 2]
    y_bin = [0 if x < mid else 1 for x in adata.obs[y_name]]
    y_bin_name = y_name + '_bin'
    adata.obs[y_bin_name] = pd.Categorical(y_bin)
    Y = adata.obs[y_bin_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    DP_diff = []
    EO_diff = []

    ACC = []

    print(f'fairness metrics wrt Si for XGBoost classifier {y_bin_name} | (Z - Zi)')

    for i in range(1, len(cats) + 1):

        Z_not_i = adata.obsm[f"Z_not_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        DP_diff_i = []
        EO_diff_i = []

        ACC_i = []

        for train, test in cv.split(Z_not_i, Y):
            Z_not_i_train = Z_not_i[train]
            Z_not_i_test = Z_not_i[test]
            Y_train = Y[train]
            Y_test = Y[test]

            Si_test = Si[test]

            estimator = clone(clf)

            estimator.fit(Z_not_i_train, Y_train, eval_set=[(Z_not_i_test, Y_test)], verbose=False)

            Y_test = pd.Series(Y_test, dtype=int)

            Y_pred = estimator.predict(Y_test)

            dp_diff = demographic_parity_difference(Y_test, Y_pred, sensitive_features=Si_test)
            DP_diff_i.append(dp_diff)
            eo_diff = equalized_odds_difference(Y_test, Y_pred, sensitive_features=Si_test)
            EO_diff_i.append(eo_diff)

            test_acc = estimator.score(Z_not_i_test, Y_test)
            ACC_i.append(test_acc)

        dp_diff = np.mean(DP_diff_i)
        DP_diff.append(dp_diff)
        eo_diff = np.mean(EO_diff_i)
        EO_diff.append(eo_diff)

        test_acc = np.mean(ACC_i)
        ACC.append(test_acc)

        print(f'i={i}: accuracy = {test_acc:.4f}, DP_diff = {dp_diff:.4f}, EO_diff = {eo_diff:.4f}')

    return ACC, DP_diff, EO_diff


def max_dim_MI_metrics(adata, cats):
    # Max Mutual Information by taking Max over Dims
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

    print('Max_Dim Mutual Information metrics')

    MI_dif = []
    H = []

    MI = []
    MI_not = []

    for i in range(1, len(cats) + 1):
        Zi = adata.obsm[f"Z_{i}"]
        Z_not_i = adata.obsm[f"Z_not_{i}"]
        Si = adata.obs[cats[i - 1] + '_idx']

        # MI

        mi = mutual_info_classif(Zi, Si, discrete_features=False)
        mi_not = mutual_info_classif(Z_not_i, Si, discrete_features=False)

        mi_max = np.max(mi)
        mi_not_max = np.max(mi_not)

        MI_dif.append(mi_max - mi_not_max)

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

        print(f"MI(Z_{i} ; S_{i}) = {mi_max:.4f},  MI((Z - Z_{i}) ; S_{i}) = {mi_not_max:.4f}")

        MI.append(mi_max)
        MI_not.append(mi_not_max)

    # MIG

    mig = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"MIG = {mig:.4f}")

    return MI, MI_not, mig


def disc_MI_metrics(adata, cats):
    # Mutual Information by discretizing Z
    # R package infotheo

    print('Z_Discretized Mutual Information metrics')

    Z = []
    Z_not = []
    S = []

    for i in range(1, len(cats) + 1):
        Zi_df = pd.DataFrame(adata.obsm[f"Z_{i}"])
        Z_not_i_df = pd.DataFrame(adata.obsm[f"Z_not_{i}"])
        Si_df = pd.DataFrame(adata.obs[cats[i - 1] + '_idx'])

        with (ro.default_converter + pandas2ri.converter).context():
            Zi_dis = ro.conversion.get_conversion().py2rpy(Zi_df)
            Z_not_i_dis = ro.conversion.get_conversion().py2rpy(Z_not_i_df)
            Si_dis = ro.conversion.get_conversion().py2rpy(Si_df)

        Z.append(infotheo.discretize(Zi_dis))
        Z_not.append(infotheo.discretize(Z_not_i_dis))
        S.append(Si_dis)

    MI = []
    MI_not = []
    MI_not_max = []

    MI_dif = []
    MI_dif_max = []
    H = []

    for i in range(1, len(cats) + 1):
        # MI

        Si = S[i-1]

        mi = infotheo.mutinformation(Z[i - 1], Si)[0]
        mi_not = infotheo.mutinformation(Z_not[i - 1], Si)[0]
        mi_not_max = max(infotheo.mutinformation(Z[j], Si)[0] for j in range(len(Z)) if j != i - 1)

        MI.append(mi)
        MI_not.append(mi_not)
        MI_not_max.append(mi_not_max)

        MI_dif.append(mi - mi_not)
        MI_dif_max.append(mi - mi_not_max)

        print(f"MI(Z_{i} ; S_{i}) = {mi:.4f},  "
              f"MI((Z - Z_{i}) ; S_{i}) = {mi_not:.4f}, "
              f"max MI((Z_j!={i}) ; S_{i}) = {mi_not_max:.4f}")

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

    # MIG, MIPG

    mig = np.mean([MI_dif_max[i] / H[i] for i in range(len(cats))])
    mipg = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"MIG = {mig:.4f}, MIPG = {mipg:.4f}")

    return MI, MI_not, MI_not_max, mig, mipg


def MP_MI_metrics(adata, cats):
    # Mixed-Pair Mutual Information Estimator
    # R package mpmi

    print('Mixed-Pair Mutual Information metrics')

    Z = []
    Z_not = []
    S = []

    for i in range(1, len(cats) + 1):
        Zi_df = pd.DataFrame(adata.obsm[f"Z_{i}"])
        Z_not_i_df = pd.DataFrame(adata.obsm[f"Z_not_{i}"])
        Si_df = pd.DataFrame(adata.obs[cats[i - 1] + '_idx'])

        with (ro.default_converter + pandas2ri.converter).context():
            Zi_r = ro.conversion.get_conversion().py2rpy(Zi_df)
            Z_not_i_r = ro.conversion.get_conversion().py2rpy(Z_not_i_df)
            Si_r = ro.conversion.get_conversion().py2rpy(Si_df)

        Z.append(Zi_r)
        Z_not.append(Z_not_i_r)
        S.append(Si_r)

    MI = []
    MI_not = []
    MI_not_max = []

    MI_dif = []
    MI_dif_max = []
    H = []

    for i in range(1, len(cats) + 1):
        # MI

        Si = S[i-1]

        mi = mpmi.mmi(Z[i - 1], Si)
        mi_not = infotheo.mutinformation(Z_not[i - 1], Si)[0]
        mi_not_max = max(infotheo.mutinformation(Z[j], Si)[0] for j in range(len(Z)) if j != i - 1)

        # mi = infotheo.mutinformation(Z[i - 1], Si)[0]
        # mi_not = infotheo.mutinformation(Z_not[i - 1], Si)[0]
        # mi_not_max = max(infotheo.mutinformation(Z[j], Si)[0] for j in range(len(Z)) if j != i - 1)

        MI.append(mi)
        MI_not.append(mi_not)
        MI_not_max.append(mi_not_max)

        MI_dif.append(mi - mi_not)
        MI_dif_max.append(mi - mi_not_max)

        print(f"MI(Z_{i} ; S_{i}) = {mi:.4f},  "
              f"MI((Z - Z_{i}) ; S_{i}) = {mi_not:.4f}, "
              f"max MI((Z_j!={i}) ; S_{i}) = {mi_not_max:.4f}")

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

    # MIG, MIPG

    mig = np.mean([MI_dif_max[i] / H[i] for i in range(len(cats))])
    mipg = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"MIG = {mig:.4f}, MIPG = {mipg:.4f}")

    return MI, MI_not, MI_not_max, mig, mipg


def Mixed_KSG_MI_metrics(adata, cats):
    # Mutual Information by Mixed_KSG
    # code from: https://github.com/wgao9/mixed_KSG/blob/master/mixed.py

    print('Mixed_KSG Mutual Information metrics')

    Z = []
    Z_not = []
    S = []

    for i in range(1, len(cats) + 1):
        Z.append(adata.obsm[f"Z_{i}"])
        Z_not.append(adata.obsm[f"Z_not_{i}"])
        S.append(adata.obs[cats[i - 1] + '_idx'])

    MI = []
    MI_not = []
    MI_not_max = []

    MI_dif = []
    MI_dif_max = []
    H = []

    for i in range(1, len(cats) + 1):
        # MI

        Si = S[i-1]

        mi = Mixed_KSG_MI(Z[i-1], Si)
        mi_not = Mixed_KSG_MI(Z_not[i-1], Si)
        mi_not_max = max(Mixed_KSG_MI(Z[j], Si) for j in range(len(Z)) if j != i - 1)

        MI.append(mi)
        MI_not.append(mi_not)
        MI_not_max.append(mi_not_max)

        MI_dif.append(mi - mi_not)
        MI_dif_max.append(mi - mi_not_max)

        print(f"MI(Z_{i} ; S_{i}) = {mi:.4f},  "
              f"MI((Z - Z_{i}) ; S_{i}) = {mi_not:.4f}, "
              f"max MI((Z_j!={i}) ; S_{i}) = {mi_not_max:.4f}")

        # entropy

        value, counts = np.unique(Si, return_counts=True)
        H.append(entropy(counts))

    # MIG, MIPG

    mig = np.mean([MI_dif_max[i] / H[i] for i in range(len(cats))])
    mipg = np.mean([MI_dif[i] / H[i] for i in range(len(cats))])

    print(f"MIG = {mig:.4f}, MIPG = {mipg:.4f}")

    return MI, MI_not, MI_not_max, mig, mipg


def Mixed_KSG_MI(x, y, k=5):
    """
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *Mixed-KSG* mutual information estimator

        Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
        y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
        k: k-nearest neighbor parameter

        Output: one number of I(X;Y)
    """
    x = np.array(x)
    y = np.array(y)

    assert len(x) == len(y), "Lists should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    N = len(x)
    if x.ndim == 1:
        x = x.reshape((N, 1))
    if y.ndim == 1:
        y = y.reshape((N, 1))
    data = np.concatenate((x, y), axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point, k + 1, p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i], 1e-15, p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i], 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], 1e-15, p=float('inf')))
        else:
            nx = len(tree_x.query_ball_point(x[i], knn_dis[i] - 1e-15, p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i], knn_dis[i] - 1e-15, p=float('inf')))
        ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny)) / N
    return ans
