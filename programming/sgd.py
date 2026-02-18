import numpy as np
import time

from typing import Optional
from dataclasses import dataclass
from model import Model, gradients, loss, scores, predict
from metrics import accuracy


@dataclass
class SGDConfig:
    learning_rate: float = 0.1
    batch_size: int = 32
    l2_reg: float = 0.0
    shuffle: bool = True
    seed: int = 0

    # threshold-based training
    max_epochs: int = 200
    target_loss: Optional[float] = None
    target_patience: int = 1
    eval_every: int = 1


@dataclass
class SGDHistory:
    train_loss: list[float]
    epochs_run: int
    time_points: list[float]
    val_acc: Optional[list[float]] = None


def train_sgd(model: Model, images: np.ndarray, labels: np.ndarray,
              cfg: SGDConfig, *,
              val_images: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None) -> SGDHistory:
    """Train multinomial logistic regression using (mini-batch) SGD until max_epochs or threshold."""
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if cfg.target_patience <= 0:
        raise ValueError("target_patience must be positive")
    
    t0 = time.perf_counter()
    time_points: list[float] = []
    val_acc_hist: Optional[list[float]] = [] if (val_images is not None and val_labels is not None) else None

    rng = np.random.default_rng(cfg.seed)
    n = labels.size
    train_loss: list[float] = []

    # Full objective on training set (data-fit + optional l2)
    def full_objective() -> float:
        obj = loss(scores(model, images), labels)
        if cfg.l2_reg > 0.0:
            obj += 0.5 * cfg.l2_reg * float(np.sum(model.weights ** 2))
        return float(obj)

    # initial objective (epoch 0)
    obj0 = full_objective()
    train_loss.append(obj0)
    time_points.append(time.perf_counter() - t0)
    if val_acc_hist is not None:
        pred_val = predict(model, val_images)
        val_acc_hist.append(accuracy(pred_val, val_labels))

    hits = 0
    if cfg.target_loss is not None and obj0 <= cfg.target_loss:
        hits = 1
        if hits >= cfg.target_patience:
            return SGDHistory(train_loss=train_loss, epochs_run=0, time_points=time_points, val_acc=val_acc_hist)

    for epoch in range(cfg.max_epochs):
        # Shuffle at epoch start (optional)
        if cfg.shuffle:
            perm = rng.permutation(n)
            images_epoch = images[perm]
            labels_epoch = labels[perm]
        else:
            images_epoch = images
            labels_epoch = labels

        # Mini-batch updates
        for start in range(0, n, cfg.batch_size):
            end = min(start + cfg.batch_size, n)
            x_batch = images_epoch[start:end]
            y_batch = labels_epoch[start:end]

            grad_w, grad_b = gradients(model, x_batch, y_batch, l2_reg=cfg.l2_reg)
            model.weights -= cfg.learning_rate * grad_w
            model.biases -= cfg.learning_rate * grad_b

        # Evaluate objective and check stopping
        if ((epoch + 1) % cfg.eval_every) == 0:
            obj = full_objective()
            train_loss.append(obj)
            time_points.append(time.perf_counter() - t0)
            if val_acc_hist is not None:
                pred_val = predict(model, val_images)
                val_acc_hist.append(accuracy(pred_val, val_labels))

            if cfg.target_loss is not None:
                if obj <= cfg.target_loss:
                    hits += 1
                else:
                    hits = 0

                if hits >= cfg.target_patience:
                    return SGDHistory(train_loss=train_loss, epochs_run=epoch + 1, time_points=time_points, val_acc=val_acc_hist)

    return SGDHistory(train_loss=train_loss, epochs_run=cfg.max_epochs, time_points=time_points, val_acc=val_acc_hist)