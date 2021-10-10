import numpy as np
from sklearn.metrics import accuracy_score, roc_curve


def accuracy(true, pred):
    return accuracy_score(true, pred)


def eer(true, pred_probs):
    fpr, tpr, thresholds = roc_curve(true, pred_probs)
    fnr = 1 - tpr

    min_rate_idx = np.argmin(np.abs(fpr - fnr))
    eer_thr = thresholds[min_rate_idx]
    eer = np.mean([fpr[min_rate_idx], fnr[min_rate_idx]])
    return eer, eer_thr


def eer_batches(true_batch, pred_probs_batch):
    return eer(true_batch.reshape(-1, 1), pred_probs_batch.reshape(-1, 1))