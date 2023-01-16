from dataset import load_data, load_data_test
from create_models import get_model_tile
from keras import layers, optimizers, losses, callbacks
from keras import Sequential

import tensorflow as tf

import argparse


def get_args():
    parser = argparse.ArgumentParser(
            prog='CIFARTileC',
            description='CIFAR 10 Tile Image Classification',
            )
    parser.add_argument('data_dir',
                        help='data folder contains train, valid, test dataset')
    parser.add_argument('cifar_weight',
                        help='pre-trained cifar model weight file')
    parser.add_argument('--backbone', default=0, choices=(0, 1), type=int,
                        help='backbones [mobilenet, resnet50]')
    parser.add_argument('--data_shape', default=(64, 64, 3),
                        help='shape of each input data')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='model fitting batch size')
    parser.add_argument('--epochs', default=2, type=int,
                        help='model fitting epoch')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='model fitting epoch')
    parser.add_argument('--subsample', default=None, type=int,
                        help='subsample dataset for local processing')
    parser.add_argument('--tpu', action='store_true',
                        help='training on TPU')
    parser.add_argument('--finetune', action='store_true',
                        help='finetune on pre-trained weight')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    if args.tpu:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
            print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        except ValueError:
            raise BaseException('ERROR: Not connected to a TPU runtime!')
            exit()

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        tpu_strategy = tf.distribute.TPUStrategy(tpu)

    (x_train, y_train), (x_valid, y_valid) = load_data(args.data_dir, args.subsample)

    def _model():
        model = get_model_tile(args.data_shape, args.cifar_weight, args.backbone)
        model.compile(
                optimizer=optimizers.Adam(args.lr),
                loss=losses.CategoricalCrossentropy(from_logits=False),
                metrics=["accuracy"]
                )
        return model

    if args.tpu:
        with tpu_strategy.scope():
            model = _model()
    else:
        model = _model()

    model.summary(expand_nested=True)

    weight_out = '{}.tile.hdf5'.format(args.cifar_weight)

    if args.finetune:
        model.load_weights(weight_out)
        print("Load weights {} ...".format(weight_out))

    model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=weight_out,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

    csv_logger = callbacks.CSVLogger('training_tile.log')

    model.fit(x_train, y_train,
              batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_valid, y_valid),
              callbacks=[model_checkpoint_callback, csv_logger])

    x_test, y_test = load_data_test(args.data_dir, args.subsample)
    model.evaluate(x_test, y_test)
