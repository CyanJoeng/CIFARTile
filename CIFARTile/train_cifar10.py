from create_models import get_model_cifar10_class
from dataset import load_cifar10
from keras import optimizers, losses, callbacks, metrics
import argparse


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


if __name__ == "__main__":
    args = get_args()

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_cifar10()

    model = get_model_cifar10_class(x_train.shape[1:], backbone=args.backbone)

    model.summary(expand_nested=True)
    print("train shape {}, {}".format(x_train.shape, y_train.shape))

    if args.checkpoint_file.endswith('.hdf5'):
        model.load_weights(args.checkpoint_file)
        print("Load weights {} ...".format(args.checkpoint_file))

    log_file_name = '{}.training.log'.format(args.checkpoint_file)
    checkpoint_file_name = args.checkpoint_file_name
    if not checkpoint_file_name.endswith('.hdf5'):
        checkpoint_file_name = checkpoint_file_name + '.hdf5'

    model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_file_name,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

    csv_logger = callbacks.CSVLogger(log_file_name)

    model.compile(
            optimizer=optimizers.Adam(args.lr),
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
            )
    model.fit(x_train, y_train,
              batch_size=args.batch_size, epochs=args.epochs, validation_data=(x_valid, y_valid),
              callbacks=[model_checkpoint_callback, csv_logger])

    model.evaluate(x_test, y_test)
