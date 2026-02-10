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
    W = np.random.randn(image_dim, num_classes)
    # Keep bises at zero for now, since I dont have a good intuition for how
    # to initialize them and they can be learned during training anywy
    b = np.zeros(num_classes, dtype=np.float64)
    return Model(weights=W, biases=b)


def scores(model: Model, images: np.ndarray) -> np.ndarray:
    return images @ model.weights + model.biases


def loss(scores: np.ndarray, labels: np.ndarray) -> float:
    log_sum_exp = np.log(np.sum(np.exp(scores), axis=1))
    # THis should be the same as using one hot vectors
    ps = scores[np.arange(labels.size), labels]
    return np.mean(log_sum_exp - ps)


def predict(model: Model, images: np.ndarray) -> np.ndarray:
    return np.argmax(scores(model, images), axis=1)
