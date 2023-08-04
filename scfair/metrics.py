from typing import Dict, List, Tuple
from sklearn.feature_selection import *

from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate

import xgboost as xgb

from scipy.stats import entropy

from fairlearn.metrics import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    for i in range(1, len(cats) + 1):

        print(f'metrics for XGBoost classifier S{i} | Z{i}')

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

        print(f'train acc: {train_score_i:.4f},  test acc: {test_score_i:.4f}')

    return acc_results


def clf_S_Z_not_metrics(adata, cats):
    # classifier Si | (Z - Zi)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)

    acc_results = []

    for i in range(1, len(cats) + 1):

        print(f'metrics for XGBoost classifier S{i} | (Z - Z{i})')

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

        print(f'train acc: {train_score_i:.4f},  test acc: {test_score_i:.4f}')

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

    for i in range(1, len(cats) + 1):

        print(f'fairness metrics wrt S{i} for XGBoost classifier {y_bin_name} | (Z - Z{i})')

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

        print(f'accuracy = {test_acc:.4f}')
        print(f'DP_diff = {dp_diff:.4f}, EO_diff = {eo_diff:.4f}')

    return ACC, DP_diff, EO_diff


def MI_metrics(adata, cats):
    # Mutual Information
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html

    print('Mutual Information metrics')

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
