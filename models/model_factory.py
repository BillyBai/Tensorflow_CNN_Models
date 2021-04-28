import tensorflow as tf
import config
from models.Vgg import Vgg
from models.ResNest import ResNest
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB4, EfficientNetB5


def get_model(model_name='ResNest50', input_shape=config.input_shape, n_classes=config.category_num,
              verbose=False, dropout_rate=0, fc_activation=None, **kwargs):
    model_name = model_name.lower()

    resnest_parameters = {
        'resnest50': {
            'blocks_set': [3, 4, 6, 3],
            'stem_width': 32,
        },
        'resnest101': {
            'blocks_set': [3, 4, 23, 3],
            'stem_width': 64,
        },
        'resnest200': {
            'blocks_set': [3, 24, 36, 3],
            'stem_width': 64,
        },
        'resnest269': {
            'blocks_set': [3, 30, 48, 8],
            'stem_width': 64,
        },
    }

    if model_name in resnest_parameters.keys():
        model = ResNest(verbose=verbose, input_shape=input_shape,
                        n_classes=n_classes, dropout_rate=dropout_rate, fc_activation=fc_activation,
                        blocks_set=resnest_parameters[model_name]['blocks_set'], radix=2, groups=1, bottleneck_width=64,
                        deep_stem=True,
                        stem_width=resnest_parameters[model_name]['stem_width'], avg_down=True, avd=True,
                        avd_first=False, **kwargs).build()
    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))
    return model


def Vgg_model():
    model = Vgg()

    return model


def ResNet50V1_model():
    model = ResNet50(
        weights=None,
        input_shape=config.input_shape,
        classes=config.category_num
    )

    return model


def ResNet50V2_model():
    model = ResNet50V2(
        weights=None,
        input_shape=config.input_shape,
        classes=config.category_num
    )

    return model


def Inception_ResNetV2_model():
    model = InceptionResNetV2(
        weights=None,
        input_shape=config.input_shape,
        classes=config.category_num
    )

    return model


def EfficientNetB4_model():
    model = EfficientNetB4(
        weights=None,
        input_shape=config.input_shape,
        classes=config.category_num
    )

    return model


def EfficientNetB5_model():
    model = EfficientNetB5(
        weights=None,
        input_shape=config.input_shape,
        classes=config.category_num
    )

    return model


def ResNet50V1_pretrain_model():
    modelPre = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=config.input_shape,
        classes=config.category_num
    )
    model = tf.keras.Sequential()
    model.add(modelPre)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(config.category_num, name='fully_connected', activation='softmax', use_bias=False))

    return model


def ResNet50V2_pretrain_model():
    modelPre = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=config.input_shape,
        classes=config.category_num
    )
    model = tf.keras.Sequential()
    model.add(modelPre)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(config.category_num, name='fully_connected', activation='softmax', use_bias=False))

    return model


def Inception_ResNetV2_pretrain_model():
    modelPre = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=config.input_shape,
        classes=config.category_num
    )
    model = tf.keras.Sequential()
    model.add(modelPre)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(config.category_num, name='fully_connected', activation='softmax', use_bias=False))

    return model


def EfficientNetB4_pretrain_model():
    modelPre = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=config.input_shape,
        classes=config.category_num
    )
    model = tf.keras.Sequential()
    model.add(modelPre)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(config.category_num, name='fully_connected', activation='softmax', use_bias=False))

    return model


def EfficientNetB5_pretrain_model():
    modelPre = EfficientNetB5(
        weights='imagenet',
        include_top=False,
        input_shape=config.input_shape,
        classes=config.category_num
    )
    model = tf.keras.Sequential()
    model.add(modelPre)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(config.category_num, name='fully_connected', activation='softmax', use_bias=False))

    return model


def get_models(model_name):
    if model_name == 'ResNet50_v1':
        return ResNet50V1_model()
    elif model_name == 'ResNet50_v1_Pretrain':
        return ResNet50V1_pretrain_model()
    elif model_name == 'ResNet50_v2':
        return ResNet50V2_model()
    elif model_name == 'ResNet50_v2_Pretrain':
        return ResNet50V2_pretrain_model()
    elif model_name == 'Vgg':
        return Vgg_model()
    elif model_name == 'Inception_ResNetV2':
        return Inception_ResNetV2_model()
    elif model_name == 'Inception_ResNetV2_Pretrain':
        return Inception_ResNetV2_pretrain_model()
    elif model_name == 'EfficientNetB4':
        return EfficientNetB4_model()
    elif model_name == 'EfficientNetB4_Pretrain':
        return EfficientNetB4_pretrain_model()
    elif model_name == 'EfficientNetB5':
        return EfficientNetB5_model()
    elif model_name == 'EfficientNetB5_Pretrain':
        return EfficientNetB5_pretrain_model()
    else:
        raise ValueError('Unrecognize model name {}'.format(model_name))


if __name__ == '__main__':
    my_model = get_models('EfficientNetB5_Pretrain')
    my_model.summary()
