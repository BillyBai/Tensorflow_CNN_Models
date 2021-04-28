import tensorflow as tf
import config


class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)


def cross_entropy_batch(y_true, y_pred, label_smoothing=0.0):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    acc = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return acc


def correct_num_batch(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    correct_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.int32))
    return correct_num


def l2_loss(model, weights=config.weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights

