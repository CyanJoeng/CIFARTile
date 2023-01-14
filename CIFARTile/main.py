from keras import layers, utils, optimizers, losses
from keras import Model, Sequential, Input

from resnet import ResNetEqualFeatureMap

from typing import Tuple
import numpy as np
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(
            prog='CIFARTileC',
            description='CIFAR 10 Tile Image Classification',
            )
    parser.add_argument('data_dir',
                        help='data folder contains train, valid, test dataset')
    parser.add_argument('--data_shape', default=(3, 64, 64),
                        help='shape of each input data')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='model fitting batch size')
    parser.add_argument('--epochs', default=2, type=int,
                        help='model fitting epoch')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)

    return args


def load_data(data_dir: str, subsample=None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    train_x = np.load(os.path.join(data_dir, "train_x.npy"))
    train_y = np.load(os.path.join(data_dir, "train_y.npy"))
    valid_x = np.load(os.path.join(data_dir, "valid_x.npy"))
    valid_y = np.load(os.path.join(data_dir, "valid_y.npy"))

    train_y = utils.to_categorical(train_y, 4)
    valid_y = utils.to_categorical(valid_y, 4)

    train_x, train_y = np.array(train_x, np.float32), np.array(train_y, np.int64)
    valid_x, valid_y = np.array(valid_x, np.float32), np.array(valid_y, np.int64)

    if subsample:
        len_train, len_valid = len(train_x), len(valid_x)
        idx_train = (np.arange(len_train) % subsample) == 0
        idx_valid = (np.arange(len_valid) % subsample) == 0
        train_x, train_y = train_x[idx_train], train_y[idx_train]
        valid_x, valid_y = valid_x[idx_valid], valid_y[idx_valid]

    print("Load data:\n\ttrain: x {}, y {}\n\tvalid: x {}, y {}".format(
        train_x.shape, train_y.shape, valid_x.shape, valid_y.shape
        ))

    return (train_x, train_y), (valid_x, valid_y)


def load_data_test(data_dir: str, subsample=None) -> Tuple[np.ndarray, np.ndarray]:
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))

    test_y = utils.to_categorical(test_y, 4)

    test_x, test_y = np.array(test_x).astype(np.float32), np.array(test_y).astype(np.int64)

    if subsample:
        len_test = len(test_x)
        idx_test = (np.arange(len_test) % subsample) == 0
        test_x, test_y = test_x[idx_test], test_y[idx_test]

    print("Load data:\n\ttest: x {}, y {}".format(
        test_x.shape, test_y.shape
        ))

    return test_x, test_y


def get_model(in_shape: tuple, data_aug: Sequential = Sequential([])) -> Model:
    resnet_feature = ResNetEqualFeatureMap()

    inputs = Input(in_shape)
    outputs = data_aug(inputs)
    outputs = resnet_feature(outputs)
    outputs = layers.GlobalAvgPool2D()(outputs)
    outputs = layers.BatchNormalization()(outputs)
    outputs = layers.Dense(100, activation='relu')(outputs)
    outputs = layers.Dropout(0.5)(outputs)
    outputs = layers.Dense(4, activation='softmax')(outputs)

    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":
    args = get_args()

    resize_scale = 2
    data_aug_layer = Sequential([
        layers.Permute((2, 3, 1)),
        layers.Resizing(args.data_shape[1] * resize_scale, args.data_shape[2] * resize_scale),
        ], name='data_aug')

    model = get_model(args.data_shape, data_aug_layer)
    model.summary()

    (x_train, y_train), (x_valid, y_valid) = load_data(args.data_dir, args.subsample)

    model.compile(
            optimizer=optimizers.RMSprop(1e-3),
            loss=losses.CategoricalCrossentropy(from_logits=False),
            metrics=["acc"]
            )
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_valid, y_valid))

    x_test, y_test = load_data_test(args.data_dir, args.subsample)
    model.evaluate(x_test, y_test)
