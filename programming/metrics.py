import numpy as np

def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    pred = np.asarray(pred).ravel()
    labels = np.asarray(labels).ravel()
    return float(np.mean(pred == labels))

def time_to_acc_threshold(time_points: list[float], acc: list[float], thr: float) -> float:
    a = np.asarray(acc, dtype=float)
    t = np.asarray(time_points, dtype=float)
    idx = np.where(a >= thr)[0]
    return float(t[idx[0]]) if idx.size else float("inf")
