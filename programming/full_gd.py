from dataclasses import dataclass
from typing import Optional
import time
import numpy as np

from model import Model, gradients, loss, scores
from metrics import accuracy


@dataclass
class GDConfig:
    learning_rate: float = 0.1
    l2_reg: float = 0.0

    # threshold-based training
    max_iters: int = 200
    target_acc: Optional[float] = None
    target_patience: int = 1


@dataclass
class GDHistory:
    train_loss: list[float]
    iters_run: int
    time_points: list[float]
    val_acc: Optional[list[float]] = None


def train_full_gd(model: Model, images: np.ndarray, labels: np.ndarray,
                  cfg: GDConfig, *,
                  val_images: Optional[np.ndarray] = None,
                  val_labels: Optional[np.ndarray] = None) -> GDHistory:
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if cfg.target_patience <= 0:
        raise ValueError("target_patience must be positive")

    if (val_images is None) != (val_labels is None):
        raise ValueError("val_images and val_labels must be both provided or both None.")

    t0 = time.perf_counter()
    time_points: list[float] = []
    val_acc_hist: Optional[list[float]] = [] if (val_images is not None) else None
    train_loss: list[float] = []

    def full_objective() -> float:
        obj = loss(scores(model, images), labels)
        if cfg.l2_reg > 0.0:
            obj += 0.5 * cfg.l2_reg * float(np.sum(model.weights ** 2))
        return float(obj)

    # Local prediction to avoid depending on model.predict (optional but robust)
    def predict_local(m: Model, X: np.ndarray) -> np.ndarray:
        return np.argmax(scores(m, X), axis=1)

    hits = 0

    # initial objective + optional validation accuracy at iter 0
    obj0 = full_objective()
    train_loss.append(obj0)
    time_points.append(time.perf_counter() - t0)

    if val_acc_hist is not None:
        pred_val = predict_local(model, val_images)
        acc0 = accuracy(pred_val, val_labels)
        val_acc_hist.append(acc0)

        # Check threshold at iter 0
        if cfg.target_acc is not None and acc0 >= cfg.target_acc:
            hits = 1
            if hits >= cfg.target_patience:
                return GDHistory(
                    train_loss=train_loss,
                    iters_run=0,
                    time_points=time_points,
                    val_acc=val_acc_hist
                )

    for k in range(cfg.max_iters):
        grad_w, grad_b = gradients(model, images, labels, l2_reg=cfg.l2_reg)
        model.weights -= cfg.learning_rate * grad_w
        model.biases -= cfg.learning_rate * grad_b

        obj = full_objective()
        train_loss.append(obj)
        time_points.append(time.perf_counter() - t0)

        if val_acc_hist is not None:
            pred_val = predict_local(model, val_images)
            acc_val = accuracy(pred_val, val_labels)
            val_acc_hist.append(acc_val)

            # Threshold stopping on validation accuracy
            if cfg.target_acc is not None:
                hits = hits + 1 if acc_val >= cfg.target_acc else 0
                if hits >= cfg.target_patience:
                    return GDHistory(
                        train_loss=train_loss,
                        iters_run=k + 1,
                        time_points=time_points,
                        val_acc=val_acc_hist
                    )

    return GDHistory(
        train_loss=train_loss,
        iters_run=cfg.max_iters,
        time_points=time_points,
        val_acc=val_acc_hist
    )
