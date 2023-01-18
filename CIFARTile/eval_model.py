from create_models import get_model_cifar10_class, get_model_tile
from dataset import load_cifar10, load_data_test
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
    parser.add_argument('model', default=0, choices=(0, 1), type=int,
                        help='backbones [CIFAR10, CIFARTile]')
    parser.add_argument('--data_folder', default=None, type=str,
                        help='data folder contains train, valid, test dataset')
    parser.add_argument('--data_shape', default=(64, 64, 3),
                        help='shape of each input data')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')
    parser.add_argument('--export', action='store_true',
                        help='export model to .h5 file')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of workers doing predict')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    if args.model == 0:
        _, _, (x_test, y_test) = load_cifar10(args.subsample)
        model = get_model_cifar10_class(x_test.shape[1:],
                                        backbone=args.backbone,
                                        pretrain=True)
    else:
        assert args.data_folder is not None
        x_test, y_test = load_data_test(args.data_folder, args.subsample)
        cifar10_weight = args.checkpoint_file[:-len('.tile.hdf5')]
        y_test = np.argmax(y_test, axis=1)
        model = get_model_tile(args.data_shape, weight_backbone=cifar10_weight,  backbone=args.backbone)

    model.summary()
    model.load_weights(args.checkpoint_file)
    model.trainable = False
    if args.export:
        export_file_name = args.checkpoint_file[:-len('.hdf5')]
        export_file_name = "{}.export.h5".format(export_file_name)
        model.save(export_file_name)
        print("Save model to ", export_file_name)

    y_pred = model.predict(x_test, workers=args.workers)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.array(y_test).reshape(-1))
    print("predict accuracy: {}\n\tModel: {}".format(accuracy, args.checkpoint_file))
