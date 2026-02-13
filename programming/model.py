from dataclasses import dataclass
import numpy as np

# from data import one_hot_encode


@dataclass
class Model:
    weights: np.ndarray  # shape (64, 10) columns are the weights w_i
    biases: np.ndarray  # shape (10,)


def init_model(image_dim: int = 64, num_classes: int = 10,
               seed: int = 0) -> Model:
    np.random.seed(seed)
    w = np.random.randn(image_dim, num_classes)
    # Keep bises at zero for now, since I dont have a good intuition for how
    # to initialize them and they can be learned during training anywy
    b = np.zeros(num_classes, dtype=np.float64)
    return Model(weights=w, biases=b)


def scores(model: Model, images: np.ndarray) -> np.ndarray:
    return images @ model.weights + model.biases


def loss(scores: np.ndarray, labels: np.ndarray) -> float:
    log_sum_exp = np.log(np.sum(np.exp(scores), axis=1))
    # THis should be the same as using sum with one hot vectors
    ps = scores[np.arange(labels.size), labels]
    return np.mean(log_sum_exp - ps)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    max_per_row = np.max(scores, axis=1, keepdims=True)
    exps = np.exp(scores - max_per_row)
    return exps / np.sum(exps, axis=1, keepdims=True)


def gradients(model: Model, images: np.ndarray, labels: np.ndarray,
              l2_reg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of mean loss wrt weights and biases (mini-batch)."""
    batch_size = labels.size
    s = scores(model, images)
    probs = softmax(s)

    ds = probs
    ds[np.arange(batch_size), labels] -= 1.0
    ds /= batch_size

    grad_w = images.T @ ds
    grad_b = np.sum(ds, axis=0)

    if l2_reg > 0.0:
        grad_w += l2_reg * model.weights  # weight decay

    return grad_w, grad_b


def predict(model: Model, images: np.ndarray) -> np.ndarray:
    return np.argmax(scores(model, images), axis=1)
