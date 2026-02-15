from dataclasses import dataclass
import numpy as np

from model import Model, gradients, loss, scores


@dataclass
class SGDConfig:
    learning_rate: float = 0.1
    epochs: int = 50
    batch_size: int = 32
    l2_reg: float = 0.0
    shuffle: bool = True
    seed: int = 0


@dataclass
class SGDHistory:
    train_loss: list[float]


def train_sgd(model: Model, images: np.ndarray, labels: np.ndarray,
              cfg: SGDConfig) -> SGDHistory:
    """Train multinomial logistic regression using (mini-batch) SGD."""
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if cfg.epochs <= 0:
        raise ValueError("epochs must be positive")

    rng = np.random.default_rng(cfg.seed)
    n = labels.size
    train_loss: list[float] = []

    obj0 = loss(scores(model, images), labels)
    if cfg.l2_reg > 0.0:
        obj0 += 0.5 * cfg.l2_reg * float(np.sum(model.weights ** 2))
    train_loss.append(float(obj0))

    for _ in range(cfg.epochs):
        if cfg.shuffle:
            perm = rng.permutation(n)
            images_epoch = images[perm]
            labels_epoch = labels[perm]
        else:
            images_epoch = images
            labels_epoch = labels

        for start in range(0, n, cfg.batch_size):
            end = min(start + cfg.batch_size, n)
            x_batch = images_epoch[start:end]
            y_batch = labels_epoch[start:end]

            grad_w, grad_b = gradients(model, x_batch, y_batch,
                                       l2_reg=cfg.l2_reg)
            model.weights -= cfg.learning_rate * grad_w
            model.biases -= cfg.learning_rate * grad_b

        train_loss.append(loss(scores(model, images), labels))

    return SGDHistory(train_loss=train_loss)


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(pred == true))
