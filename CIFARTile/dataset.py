import os
import numpy as np
from typing import Tuple

from tensorflow.keras import utils, datasets


def subsample_dataset(data_list, subsample):
    def sample(data):
        _len = len(data)
        _idx = (np.arange(_len) % subsample) == 0
        return data[_idx]

    return [sample(data) for data in data_list]


def image_transformation(arr):
    min = arr.min()
    max = arr.max()
    arr = (arr - min) / (max - min)

    arr = np.transpose(arr, (0, 2, 3, 1))
    arr = arr * 255
    return arr


def load_data(data_dir: str, subsample=None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    train_x = np.load(os.path.join(data_dir, "train_x.npy"))
    train_y = np.load(os.path.join(data_dir, "train_y.npy"))
    valid_x = np.load(os.path.join(data_dir, "valid_x.npy"))
    valid_y = np.load(os.path.join(data_dir, "valid_y.npy"))

    train_x = image_transformation(train_x)
    valid_x = image_transformation(valid_x)

    train_y = utils.to_categorical(train_y, 4)
    valid_y = utils.to_categorical(valid_y, 4)

    train_x, train_y = np.array(train_x, np.uint8), np.array(train_y, np.int64)
    valid_x, valid_y = np.array(valid_x, np.uint8), np.array(valid_y, np.int64)

    if subsample:
        train_x, train_y, valid_x, valid_y = subsample_dataset([train_x, train_y, valid_x, valid_y], subsample)

    print("Load data:\n\ttrain: x {}, y {}\n\tvalid: x {}, y {}".format(
        train_x.shape, train_y.shape, valid_x.shape, valid_y.shape
        ))

    return (train_x, train_y), (valid_x, valid_y)


def load_data_test(data_dir: str, subsample=None) -> Tuple[np.ndarray, np.ndarray]:
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))

    test_x = image_transformation(test_x)
    test_y = utils.to_categorical(test_y, 4)

    test_x, test_y = np.array(test_x).astype(np.uint8), np.array(test_y).astype(np.int64)

    if subsample:
        test_x, test_y = subsample_dataset([test_x, test_y], subsample)

    print("Load data:\n\ttest: x {}, y {}".format(
        test_x.shape, test_y.shape
        ))

    return test_x, test_y


def load_cifar10(subsample=None):
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    if subsample:
        x_train, y_train, x_test, y_test = subsample_dataset([x_train, y_train, x_test, y_test], subsample)

    valid_idx = (np.arange(len(y_train)) % 5) == 0
    train_idx = (np.arange(len(y_train)) % 5) != 0

    x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    print("x_shape {}, y_shape {}".format(x_train.shape, y_train.shape))
    print("        {},         {}".format(x_test.shape, y_test.shape))
    print("        {},         {}".format(x_valid.shape, y_valid.shape))

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
