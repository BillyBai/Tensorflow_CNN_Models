import tensorflow as tf
import config
from models.model_factory import get_models
from utils.data_utils import train_iterator, test_iterator
from utils.train_utils import CosineDecayWithWarmUP, cross_entropy_batch, correct_num_batch, l2_loss
from test import test_all_image_10_crop
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@tf.function
def train_step(images, labels, model, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        cross_entropy = cross_entropy_batch(labels, prediction, label_smoothing=config.label_smoothing)
        l2 = l2_loss(model)
        loss = cross_entropy + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return cross_entropy, prediction


@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    cross_entropy = cross_entropy_batch(labels, prediction)
    return cross_entropy, prediction


def train(train_iterations_per_epoch, train_data_iterator, epoch_num, train_num, model, optimizer, f):
    train_sum_cross_entropy = 0
    train_sum_correct_num = 0
    for step in range(train_iterations_per_epoch):
        images, labels = train_data_iterator.next()
        cross_entropy, prediction = train_step(images, labels, model, optimizer=optimizer)
        correct_num = correct_num_batch(labels, prediction)
        train_sum_correct_num += correct_num
        train_sum_cross_entropy += cross_entropy * config.batch_size

        print("Epoch: {}/{}, step: {}/{}, cross entropy loss: {:.5f}, "
              "l2 loss: {:.5f}, accuracy: {:.5f}".format(epoch_num + 1,
                                                         config.epoch_num,
                                                         step + 1,
                                                         train_iterations_per_epoch,
                                                         cross_entropy,
                                                         l2_loss(model),
                                                         correct_num / config.batch_size))
    print('-' * 100)
    message = "Train epoch: {}/{}, cross entropy loss: {:.5f}, " \
              "l2 loss: {:.5f}, accuracy: {:.5f}".format(epoch_num + 1,
                                                         config.epoch_num,
                                                         train_sum_cross_entropy / train_num,
                                                         l2_loss(model),
                                                         train_sum_correct_num / train_num)
    print(message)
    f.write(message + '\n')

    train_loss[epoch_num] = train_sum_cross_entropy / train_num + l2_loss(model)
    train_accuracy[epoch_num] = train_sum_correct_num / train_num


def test(epoch_num, model, f):
    test_data_iterator, test_num = test_iterator()
    test_iterations_per_epoch = int(test_num / config.batch_size) + 1

    test_sum_cross_entropy = 0
    test_sum_correct_num = 0
    for i in range(test_iterations_per_epoch):
        images, labels = test_data_iterator.next()
        cross_entropy, prediction = test_step(model, images, labels)
        correct_num = correct_num_batch(labels, prediction)
        test_sum_correct_num += correct_num
        test_sum_cross_entropy += cross_entropy * config.batch_size

    message = "Test epoch: {}/{}, cross entropy loss: {:.5f}, " \
              "l2 loss: {:.5f}, accuracy: {:.5f}".format(epoch_num + 1,
                                                         config.epoch_num,
                                                         test_sum_cross_entropy / test_num,
                                                         l2_loss(model),
                                                         test_sum_correct_num / test_num)
    print(message)
    f.write(message + '\n')
    print('-' * 100)

    test_loss[epoch_num] = test_sum_cross_entropy / test_num + l2_loss(model)
    test_accuracy[epoch_num] = test_sum_correct_num / test_num


def main():
    # load data & config
    train_data_iterator, train_num = train_iterator()
    train_iterations_per_epoch = int(train_num / config.batch_size)
    warm_iterations = train_iterations_per_epoch

    # get model
    model = get_models(config.model)
    model.build(input_shape=(None,) + config.input_shape)
    model.summary()

    # set learning rate & optimizer
    learning_rate = CosineDecayWithWarmUP(initial_learning_rate=config.initial_learning_rate,
                                          decay_steps=config.epoch_num * train_iterations_per_epoch - warm_iterations,
                                          alpha=config.minimum_learning_rate,
                                          warm_up_step=warm_iterations)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # train
    with open(config.log_file, 'w') as f:
        for epoch_num in range(config.epoch_num):
            f.write('-' * 100 + '\n')
            train(train_iterations_per_epoch, train_data_iterator, epoch_num, train_num, model, optimizer, f)
            test(epoch_num, model, f)

        f.write('-' * 100 + '\n')
        f.write('-' * 100 + '\n')
        message = test_all_image_10_crop(model)
        f.write(message + '\n')
        f.write('-' * 100)

    model.save_weights(config.save_weight_file, save_format='h5')

    # plt figure loss and accuracy
    plt.figure()
    plt.plot(range(1, config.epoch_num + 1), train_loss, label='train_loss')
    plt.plot(range(1, config.epoch_num + 1), test_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 5)
    plt.title(config.train_title + '_loss')
    plt.savefig(config.plt_loss_file)
    plt.show()

    plt.figure()
    plt.plot(range(1, config.epoch_num + 1), train_accuracy, label='train_accuracy')
    plt.plot(range(1, config.epoch_num + 1), test_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(config.train_title + '_accuracy')
    plt.savefig(config.plt_accuracy_file)
    plt.show()


if __name__ == '__main__':
    train_loss = [0 for i in range(config.epoch_num)]
    test_loss = [0 for i in range(config.epoch_num)]
    train_accuracy = [0 for i in range(config.epoch_num)]
    test_accuracy = [0 for i in range(config.epoch_num)]
    main()
