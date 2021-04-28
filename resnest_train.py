import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from models.model_factory import get_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


PATH = './data/MyData'
train_dir = os.path.join(PATH, 'TrainData')
val_dir = os.path.join(PATH, 'TestData')

BATCH_SIZE = 8
EPOCHS = 50
IMG_H = 224
IMG_W = 224
model_name = 'ResNest50'
input_shape = [224, 224, 3]
n_classes = 40
fc_activation = 'softmax'


def main():
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255
        # rotation_range=40,  # 旋转范围
        # width_shift_range=0.1,  # 水平平移范围
        # height_shift_range=0.1,  # 垂直平移范围
        # shear_range=0.1,  # 剪切变换的程度
        # zoom_range=0.1,  # 剪切变换的程度
        # horizontal_flip=True,  # 水平翻转
    )

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        shuffle=True,
        target_size=(IMG_H, IMG_W),
    )

    total_train = train_data_gen.n

    val_image_generator = ImageDataGenerator(rescale=1. / 255)

    val_data_gen = val_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=val_dir,
        shuffle=True,
        target_size=(IMG_H, IMG_W),
    )

    total_val = val_data_gen.n

    model = get_model(model_name=model_name, input_shape=input_shape, n_classes=n_classes,
                      verbose=True, fc_activation=fc_activation, using_cb=False)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=val_data_gen,
                        validation_steps=total_val // BATCH_SIZE,
                        callbacks=[reduce_lr])

    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    plt.figure()
    plt.plot(range(EPOCHS), train_loss, label='train_loss')
    plt.plot(range(EPOCHS), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.figure()
    plt.plot(range(EPOCHS), train_accuracy, label='train_accuracy')
    plt.plot(range(EPOCHS), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    main()

