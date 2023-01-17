from create_models import get_model_cifar10_class
from dataset import load_cifar10
from keras import Model
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
    parser.add_argument('--show', action='store_true',
                        help='show the feature map')
    parser.add_argument('--deeper', action='store_true',
                        help='using deeper fit_to block weight')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    _, _, (x_test, y_test) = load_cifar10(args.subsample)

    base_model = get_model_cifar10_class(x_test.shape[1:],
                                         backbone=args.backbone,
                                         pretrain=True,
                                         deeper_fit_to=args.deeper
                                         )

    base_model.load_weights(args.checkpoint_file)
    base_model.trainable = False

    feature_model = Model(base_model.layers[0].input, base_model.layers[0].output)
    feature_model.summary()

    out_img = feature_model.predict(x_test[:10], workers=args.workers)
    for img in out_img:
        print(img.shape, np.max(img))
    np.save("log/cifar_model_pred_feature.npy", out_img)

    if args.show:
        plt.figure(figsize=(12, 8))
        idx = 1
        for img in out_img:
            plt.subplot(4, 5, idx)
            plt.imshow(img)
            idx = idx + 1

        for c in range(3):
            plt.subplot(4, 5, idx)
            plt.imshow(x_test[0][:, :, c], cmap='gray')
            idx = idx + 1

        plt.subplot(4, 5, idx)
        plt.imshow(x_test[0])
        idx = idx + 2

        for c in range(3):
            plt.subplot(4, 5, idx)
            plt.imshow(out_img[0][:, :, c], cmap='gray')
            idx = idx + 1
        plt.show()
