import tensorflow as tf
from tensorflow.keras import layers
import config


def Vgg():
    model = tf.keras.models.Sequential([
        layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(config.category_num, activation='softmax')
    ])

    return model


if __name__ == '__main__':
    model = Vgg()
    model.build(input_shape=config.input_shape)
    model.summary()

