from create_models import get_model_cifar10_class
from dataset import load_cifar10
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(
            prog='CIFAR10',
            description='CIFAR 10 Image Classification',
            )
    parser.add_argument('checkpoint_file',
                        help='path to save the model file')
    parser.add_argument('backbone', default=0, choices=(0, 1), type=int,
                        help='backbones [mobilenet, resnet50]')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of workers doing predict')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    _, _, (x_test, y_test) = load_cifar10(args.subsample)

    base_model = get_model_cifar10_class(x_test.shape[1:],
                                         backbone=args.backbone,
                                         pretrain=True)

    base_model.load_weights(args.checkpoint_file)
    base_model.trainable = False

    base_model.summary()

    y_pred = base_model.predict(x_test, workers=args.workers)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.array(y_test).reshape(-1))
    print("predict accuracy: {}\n\tModel: {}".format(accuracy, args.checkpoint_file))
