from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass


@dataclass
class ImageData:
    images_train: np.ndarray
    images_val: np.ndarray
    images_test: np.ndarray
    labels_train: np.ndarray
    labels_val: np.ndarray
    labels_test: np.ndarray
    count: int


def load_data(train_size=0.7, val_size=0.15, test_size=0.15,
              random_state=0) -> ImageData:
    # Basic validations
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("train_size, val_size and test_size must be > 0.")

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"sizes must sum to 1.0 (got {total}).")

    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    n = len(X)

    # split train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=y
    )

    # split temp into val vs test
    test_ratio_within_temp = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio_within_temp,
        random_state=random_state,
        shuffle=True,
        stratify=y_temp
    )

    return ImageData(X_train, X_val, X_test, y_train, y_val, y_test, n)
