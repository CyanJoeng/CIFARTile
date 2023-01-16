from create_models import get_model_cifar10_class
from dataset import subsample_dataset
from keras import datasets, optimizers, losses, callbacks, metrics
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
            prog='CIFARTileC',
            description='CIFAR 10 Tile Image Classification',
            )
    parser.add_argument('checkpoint_file', type=str,
                        help='path to save the model file')
    parser.add_argument('--backbone', default=0, choices=(0, 1), type=int,
                        help='backbones [mobilenet, resnet50]')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='model fitting batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='model fitting epoch')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='model fitting epoch')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')
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

    model = get_model_cifar10_class(x_train.shape[1:], backbone=args.backbone)

    model.summary(expand_nested=True)
    print("train shape {}, {}".format(x_train.shape, y_train.shape))

    if args.checkpoint_file.endswith('.hdf5'):
        model.load_weights(args.checkpoint_file)
        print("Load weights {} ...".format(args.checkpoint_file))

    model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=args.checkpoint_file
            if args.checkpoint_file.endswith('.hdf5') else
            args.checkpoint_file + '.hdf5',
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

    csv_logger = callbacks.CSVLogger('training_cifar10.log')

    model.compile(
            optimizer=optimizers.Adam(args.lr),
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
    model.fit(x_train, y_train,
              batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_valid, y_valid),
              callbacks=[model_checkpoint_callback, csv_logger])

    model.evaluate(x_test, y_test)
