from create_models import get_model_cifar10_class
from dataset import subsample_dataset
from keras import datasets, utils, Model
import argparse
import numpy as np

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
            prog='CIFARTileC',
            description='CIFAR 10 Tile Image Classification',
            )
    parser.add_argument('checkpoint_file',
                        help='path to save the model file')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='model fitting batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='model fitting epoch')
    parser.add_argument('--lr', default=0.001, type=int,
                        help='model fitting epoch')
    parser.add_argument('--backbone', default=0, choices=(0, 1), type=int,
                        help='backbones [mobilenet, resnet50]')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of workers doing predict')
    args = parser.parse_args()
    args = parser.parse_args()

    return args


IMG_SIZE = 224

if __name__ == "__main__":
    args = get_args()

    (x_train, y_train), (x_valid, y_valid) = datasets.cifar10.load_data()

    if args.subsample:
        x_train, y_train, x_valid, y_valid = subsample_dataset([x_train, y_train, x_valid, y_valid], args.subsample)

    test_idx = (np.arange(len(y_train)) % 5) == 0
    train_idx = (np.arange(len(y_train)) % 5) != 0

    x_test, y_test = x_train[test_idx], y_train[test_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]
    print("x_shape {}, y_shape {}".format(x_train.shape, y_train.shape))
    print("        {},         {}".format(x_valid.shape, y_valid.shape))
    print("        {},         {}".format(x_test.shape, y_test.shape))

    base_model = get_model_cifar10_class(x_train.shape[1:], backbone=args.backbone, pretrain=True)

    # utils.plot_model(base_model, expand_nested=True)

    print(base_model.inbound_nodes)

    base_model.load_weights(args.checkpoint_file)
    base_model.trainable = False

    base_model.summary()

    y_pred = base_model.predict(x_test)
    print("predict accuracy: ", np.mean(np.argmax(y_pred, axis=1) == np.array(y_test).reshape(-1)))

    # pre = Model(base_model.layers[0].input, base_model.layers[0].output)
    # backbone = Model(base_model.layers[1].input, base_model.layers[1].output)
    # model = Model(pre.input, backbone.call(pre.output))

    feature_model = Model(base_model.layers[0].input, base_model.layers[0].output)

    out_img = feature_model.predict(x_test[:10], workers=args.workers)
    for img in out_img:
        print(img.shape, np.max(img))
    np.save("log/cifar_model_pred_feature.npy", out_img)
