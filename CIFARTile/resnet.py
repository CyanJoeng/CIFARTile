from tensorflow import keras
from keras import layers, utils, optimizers, losses
from keras import Model, Sequential, Input
# from keras import applications

from typing import List


class ResBlock(Model):
    def __init__(self, kernel_size: tuple, strides: int, name: str):
        super(ResBlock, self).__init__()
        first_kernel, last_kernel = kernel_size
        self._name = name
        self._str = strides

        equal_block = [
                layers.BatchNormalization(),
                layers.Conv2D(
                    first_kernel, kernel_size=3, strides=strides,
                    padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(
                    first_kernel, kernel_size=3, padding='same',
                    activation='relu')
                ]
        unequal_block = [
                layers.BatchNormalization(),
                layers.Conv2D(
                    first_kernel, kernel_size=1, strides=strides,
                    activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(
                    first_kernel, kernel_size=3, padding='same',
                    activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(
                    last_kernel, kernel_size=1, activation='relu'),
                ]

        is_equal = first_kernel == last_kernel

        self._layer = Sequential(equal_block if is_equal else unequal_block)

        self._identity_shortcut = Sequential(
                [] if is_equal and strides == 1 else [
                    layers.BatchNormalization(),
                    layers.Conv2D(
                        last_kernel, kernel_size=1, strides=strides,
                        activation='relu')
                    ])

        # print("name {}, kernel_size {}, strides {}, name {}".format(
        #     self._name, kernel_size, strides, name))

    def call(self, inputs):

        layer_out = self._layer(inputs)
        identity_out = self._identity_shortcut(inputs)
        # print("name {}, stride {}, layer_out {}, iden_out {}".format(
        #     self._name, self._str, layer_out.shape, identity_out.shape))

        outputs = layers.Add()([layer_out, identity_out])

        # print("name {}, input size {}, out {}".format(
        #     self._name, inputs.shape, outputs.shape))

        return outputs


class ResNetX(Model):

    def __init__(self, kernels, blocks, K: int = 1000):
        super().__init__()
        self.kernels = kernels
        self.blocks = blocks
        _1_k = kernels[0][0]

        feature_only = True if K == 0 else False

        self.conv1 = Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(
                _1_k, 7, strides=2, padding='same', activation='relu'),
            ], name="conv1")

        self.res_layers = Sequential([
            self._layer(id_layer, kernels, block, "layer{}".format(id_layer))
            for id_layer, (kernels, block) in enumerate(zip(self.kernels, self.blocks))
            ], name="convs")

        self.classifier = Sequential([
            layers.GlobalAvgPool2D(),
            layers.BatchNormalization(),
            layers.Dense(K),
            layers.Softmax(),
            ]) if not feature_only else Sequential()

    def _layer(self, id_layer, out_kernel, num_block, name):

        # print("\nlayer {} out_kernel {} has {} blocks".format(id_layer, out_kernel, num_block))

        first_pooling = [layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')]
        residuals = [
                ResBlock(out_kernel,
                         2 if id_layer != 0 and block_n == 0 else 1,
                         name="{}_block{}".format(name, block_n))
                for block_n in range(num_block)]

        blocks = residuals
        if id_layer == 0:
            blocks = first_pooling + residuals

        return Sequential(blocks, name=name)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        # print("\nconv1 {} -> {}\n".format(inputs.shape, x1.shape))

        x2 = self.res_layers(x1)
        assert x1 is not None and x2 is not None
        # print("res layers {} -> {}\n".format(x1.shape, x2.shape))
        x3 = self.classifier(x2)
        assert x3 is not None
        # print("out layer {} -> {}\n".format(x2.shape, x3.shape))
        return x3


def ResNet18(first_k, out_classes):
    # kernels = [(self._1_k, self._1_k), (128, 128), (256, 256), (512, 512)]
    kernels = [(first_k * i, first_k * i) for i in range(1, 5)]
    blocks = [2, 2, 2, 2]
    return ResNetX(kernels, blocks, out_classes)


def ResNetEqualFeatureMap(first_k: int = 64, blocks: List[int] = [2, 2, 2, 2]):
    # kernels = [(self._1_k, self._1_k), (128, 128), (256, 256), (512, 512)]
    kernels = [(first_k * i, first_k * i) for i in range(1, 5)]
    return ResNetX(kernels, blocks, 0)
