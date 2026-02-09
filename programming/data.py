from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.datasets import load_digits


@dataclass
class ImageData:
    images_train: np.ndarray
    images_test: np.ndarray
    labels_train: np.ndarray
    count: int
    labels_test: Optional[np.ndarray] = None

# TODO: Normalize the pixel values to the range [0,1].

def load_data(test_size: float = 0.2, random_state: int = 0) -> ImageData:
    digits = load_digits()
    images = digits.data  # shape (1797, 64)
    labels = digits.target  # shape (1797,)
    count = len(images)

    # We want to shuffle the data because otherwise the training set always
    # contains the same digits and the test set contains the same digits,
    # which is not good for training and testing.
    np.random.seed(random_state)
    indices = np.random.permutation(count)
    images = images[indices]
    labels = labels[indices]

    # Split into training and test sets with the given proportions
    split_index = int(count * test_size)
    images_test = images[:split_index]
    labels_test = labels[:split_index]
    images_train = images[split_index:]
    labels_train = labels[split_index:]

    return ImageData(
        images_train=images_train,
        images_test=images_test,
        labels_train=labels_train,
        labels_test=labels_test,
        count=count
    )
