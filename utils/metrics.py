import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    cohen_kappa_score,
    confusion_matrix,
)
from scipy.optimize import linear_sum_assignment


def per_class_accuracy(y_true, y_pred):
    """Calculate accuracy for each class."""
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]
        y_c_pred = y_pred[np.nonzero(y_true == c)]
        ca.append(metrics.accuracy_score(y_c, y_c_pred))
    return np.array(ca)


def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm (linear sum assignment).

    Returns:
        Tuple of (acc, new_predict, mapping, purity, kappa, nmi, ari, ami, fmi, per_class_acc)
    """
    y_true = torch.tensor(y_true) - torch.min(torch.tensor(y_true))
    l1 = list(set(y_true.tolist()))
    num_class1 = len(l1)

    y_pred = torch.tensor(y_pred)
    l2 = list(set(y_pred.tolist()))
    num_class2 = len(l2)

    fill_idx = 0
    if num_class1 != num_class2:
        for i in l1:
            if i not in l2:
                y_pred[fill_idx] = i
                fill_idx += 1
    l2 = list(set(y_pred.tolist()))
    if num_class1 != len(l2):
        print('Error: class number mismatch')
        return

    cost = torch.zeros((num_class1, len(l2)), dtype=torch.int32)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    row_ind, col_ind = linear_sum_assignment(-cost.numpy())
    new_predict = torch.zeros(len(y_pred))
    mapping = {}
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        ai = [idx for idx, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
        mapping[c2] = c

    y_true_np = y_true.cpu().numpy()
    new_predict_np = new_predict.cpu().numpy()

    acc = metrics.accuracy_score(y_true_np, new_predict_np)
    ca = per_class_accuracy(y_true_np, new_predict_np)
    matrix = confusion_matrix(y_true_np, new_predict_np)
    purity = np.sum(np.max(matrix, axis=0)) / np.sum(matrix)
    ka = cohen_kappa_score(y_true_np, new_predict_np)
    nmi = normalized_mutual_info_score(y_true_np, new_predict_np)
    ami = adjusted_mutual_info_score(y_true_np, new_predict_np)
    ari = adjusted_rand_score(y_true_np, new_predict_np)
    fmi = fowlkes_mallows_score(y_true_np, new_predict_np)

    return acc, new_predict, mapping, purity, ka, nmi, ari, ami, fmi, ca


def evaluate(y_true, y_pred):
    """Evaluate clustering performance and print all metrics."""
    acc, _, mapping, purity, kappa, nmi, ari, ami, fmi, ca = clustering_accuracy(y_true, y_pred)
    print(f'acc {acc:.4f}, nmi {nmi:.4f}, ami {ami:.4f}, '
          f'ari {ari:.4f}, fmi {fmi:.4f}, kappa {kappa:.4f}, purity {purity:.4f}')
    return acc, mapping, ca
