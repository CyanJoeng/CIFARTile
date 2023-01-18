from keras import Input, Model, Sequential
from keras import layers, applications


def get_model_cifar10_class(in_shape, backbone=0, pretrain=False, classes=10, data_aug=None,
                            backbone_in_shape=(224, 224, 3), deeper_fit_to=False) -> Model:

    backbones = [
            applications.MobileNet,
            applications.ResNet50,
            ]
    backbone_model = backbones[backbone]

    print('Backbone: ', backbone_model)

    if not pretrain:
        base_model = backbone_model(input_shape=backbone_in_shape, include_top=False)
    else:
        base_model = backbone_model(input_shape=backbone_in_shape, weights='imagenet', include_top=False)
    # base_model.trainable = False

    inputs = Input(in_shape, name='input')

    def pre_process(height, width, deconvs=2):
        def _block(id):
            deep = [
                    layers.Conv2D(3, 3, padding='same', activation='relu'),
                    layers.BatchNormalization(),
                    layers.Conv2D(6, 3, padding='same', activation='relu'),
                    layers.BatchNormalization(),
                    layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='relu'),
                    layers.BatchNormalization()
                    ]
            shallow = [
                    layers.Conv2DTranspose(3, 3, strides=2, padding='same', activation='relu'),
                    layers.BatchNormalization()
                    ]

            return Sequential(deep if deeper_fit_to else shallow, name='fit_block{}'.format(id))

        return Sequential([_block(id) for id in range(deconvs)] + [
            layers.Resizing(height, width, interpolation="nearest")
            ], name='fit_to')

    def new_top(classes=10):
        return Sequential([
            layers.GlobalAvgPool2D(),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax'),
            ], name='top')

    return Sequential([
        inputs,
        pre_process(backbone_in_shape[0], backbone_in_shape[1]),
        base_model,
        new_top(classes)
        ])


def get_model_tile(in_shape: tuple, weight_backbone=None, backbone=0) -> Model:

    backbone_in_shape = (224 * 2, 224 * 2, 3)
    base_model = get_model_cifar10_class(in_shape, backbone=backbone, backbone_in_shape=backbone_in_shape)
    if weight_backbone is not None:
        base_model.load_weights(weight_backbone)
        base_model.trainable = False

    pre = Model(base_model.layers[0].input, base_model.layers[0].output)
    backbone = Model(base_model.layers[1].input, base_model.layers[1].output)
    model = Model(pre.input, backbone.call(pre.output))

    feat_map_shape = model.output_shape

    def new_top():

        def blend(kernels, idx):
            return Sequential([
                layers.Conv1D(kernels * 2, 3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv1D(kernels * 2, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv1D(kernels, 3, padding='same', activation='relu'),
                layers.MaxPooling1D(),
                layers.BatchNormalization(),
                ], name='blend{}'.format(idx))

        return Sequential([
            layers.Reshape((-1, feat_map_shape[-1])),
            layers.Permute((2, 1)),

            blend(32, idx=0),
            blend(64, idx=1),
            blend(128, idx=2),

            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(4, activation='softmax'),
            ], name='new_top')

    inputs = Input(in_shape)
    outputs = model(inputs)
    outputs = new_top()(outputs)
    model = Model(inputs, outputs)

    return model
