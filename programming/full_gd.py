from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from model import Model, gradients, loss, scores


@dataclass
class GDConfig:
    learning_rate: float = 0.1
    iterations: int = 200
    l2_reg: float = 0.0


@dataclass
class OptimHistory:
    train_loss: list[float]


def train_full_gd(model: Model, images: np.ndarray, labels: np.ndarray,
                  cfg: GDConfig) -> OptimHistory:
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.iterations <= 0:
        raise ValueError("iterations must be positive")

    train_loss: list[float] = []

    obj0 = loss(scores(model, images), labels)
    if cfg.l2_reg > 0.0:
        obj0 += 0.5 * cfg.l2_reg * float(np.sum(model.weights ** 2))
    train_loss.append(float(obj0))

    for _ in range(cfg.iterations):
        grad_w, grad_b = gradients(model, images, labels, l2_reg=cfg.l2_reg)
        model.weights -= cfg.learning_rate * grad_w
        model.biases -= cfg.learning_rate * grad_b

        obj = loss(scores(model, images), labels)
        if cfg.l2_reg > 0.0:
            obj += 0.5 * cfg.l2_reg * float(np.sum(model.weights ** 2))
        train_loss.append(float(obj))

    return OptimHistory(train_loss=train_loss)
