import tensorflow as tf
import config
import numpy as np
import json
import time
from models.model_factory import get_models
from utils.data_utils import load_image, test_10_crop_iterator
from utils.train_utils import cross_entropy_batch, l2_loss
from utils.augment_utils import test_10_crop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = './result/weight/Inception_ResNetV2_Pretrain_Tricks.h5'
image_path = './data/MyData/TestData/01_Dry_Box/01_Dry_Box_14.jpg'


def test_one_image(model):
    # show
    img, _ = load_image(tf.constant(image_path), 0)
    prediction = model(np.array([img]), training=False)
    label = np.argmax(prediction)

    with open('data/label_to_content.json', 'r') as f:
        begin_time = time.time()
        label_to_content = f.readline()
        label_to_content = json.loads(label_to_content)

        print('-' * 100)
        print('Test one image:')
        print('image: {}\nclassification: {}\nconfidence: {:.4f}'.format(image_path,
                                                                         label_to_content[str(label)],
                                                                         prediction[0, label]))
        end_time = time.time()
        run_time = end_time - begin_time
        print('run time:', run_time)
        print(np.around(np.squeeze(prediction), decimals=4))
        print('-' * 100)


@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    prediction = tf.reduce_mean(prediction, axis=0)
    cross_entropy = cross_entropy_batch([labels], [prediction])
    return cross_entropy, prediction


def test(model):
    test_data_iterator, test_num = test_10_crop_iterator()

    test_sum_cross_entropy = 0
    test_sum_correct_num = 0
    for i in range(test_num):
        images, labels = test_data_iterator.next()
        cross_entropy, prediction = test_step(model, images, labels)

        test_sum_cross_entropy += cross_entropy * config.batch_size
        if np.argmax(prediction) == np.argmax(labels):
            test_sum_correct_num += 1
    message = "Test 10 crop: cross entropy loss: {:.5f}, " \
              "l2 loss: {:.5f}, accuracy: {:.5f}".format(test_sum_cross_entropy / test_num,
                                                         l2_loss(model),
                                                         test_sum_correct_num / test_num)
    print('-' * 100)
    print(message)
    print('-' * 100)
    return message


def test_all_image_10_crop(model):
    # show
    message = test(model)
    return message


def test_one_image_10_crop(model):
    # show
    img, _ = load_image(tf.constant(image_path), 0)
    img = test_10_crop(img)
    prediction = model(img, training=False)
    prediction = tf.reduce_mean(prediction, axis=0)
    label = np.argmax(prediction)

    with open('data/label_to_content.json', 'r') as f:
        begin_time = time.time()
        label_to_content = f.readline()
        label_to_content = json.loads(label_to_content)

        print('-' * 100)
        print('Test one image 10 crop:')
        print('image: {}\nclassification: {}\nconfidence: {:.4f}'.format(image_path,
                                                                         label_to_content[str(label)],
                                                                         prediction[label]))
        end_time = time.time()
        run_time = end_time - begin_time
        print('run time:', run_time)
        print(np.around(np.squeeze(prediction), decimals=4))
        print('-' * 100)




def main():
    # get model
    print("Loading model...")
    model = get_models("Inception_ResNetV2_Pretrain")
    model.build(input_shape=(None,) + config.input_shape)
    model.load_weights(model_path)
    test_one_image(model)
    test_one_image_10_crop(model)


if __name__ == '__main__':
    main()
