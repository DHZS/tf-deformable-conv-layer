"""Deformable Convolutional Networks
"""
import numpy as np
import tensorflow as tf
from nets.classification_net import ClassificationNet

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
tf.enable_eager_execution(conf)

SEED = 1234
tf.set_random_seed(SEED)


NUM_CLASS = 10
IMG_SHAPE = [28, 28]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data('/data/ajy/datasets/MNIST/mnist.npz')
# scale to (-1, 1), shape is (28, 28, 1)
x_train, x_test = [(np.expand_dims(i / 127.5 - 1, axis=-1)).astype(np.float32) for i in [x_train, x_test]]
y_train, y_test = tf.one_hot(y_train, depth=NUM_CLASS), tf.one_hot(y_test, depth=NUM_CLASS)


def get_dataset(batch_size, x, y, map_fn, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size * 10).map(map_fn, num_parallel_calls=2).batch(batch_size).prefetch(1)
    return dataset


def distorted_image_fn(image, label):
    # random rotate
    # 80% ->(-30°, 30°), 20%->(-90°,-30°)&(30°,90°)
    tf.set_random_seed(SEED)
    small_angle = tf.cast(tf.random_uniform([1], maxval=1.) <= 0.8, tf.int32)
    angle = tf.random_uniform([1], minval=0, maxval=30, dtype=tf.int32) * small_angle + \
            tf.random_uniform([1], minval=30, maxval=90, dtype=tf.int32) * (1 - small_angle)
    negative = -1 + 2 * tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)
    angle = tf.to_float(negative * angle)
    rotated_image = tf.contrib.image.rotate(image, angle * 3.1415926 / 180)
    return rotated_image, label


def distorted_image_test_fn(image, label):
    # random rotate
    # (-135°, 135°)
    tf.set_random_seed(SEED)
    angle = tf.random_uniform([1], minval=0, maxval=135, dtype=tf.int32)
    negative = -1 + 2 * tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)
    angle = tf.to_float(negative * angle)
    rotated_image = tf.contrib.image.rotate(image, angle * 3.1415926 / 180)
    return rotated_image, label


def main():
    batch_size = 16

    dataset = get_dataset(batch_size, x_train, y_train, distorted_image_fn, repeat=True)
    model = ClassificationNet(num_class=NUM_CLASS)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    global_step = tf.train.get_or_create_global_step()

    for i, (rotated_image, label) in enumerate(dataset, start=1):
        global_step.assign_add(1)
        loss, prediction= model.train(optimizer, rotated_image, label)
        acc = model.accuracy(prediction, label)

        # test
        if i % 1000 == 0:
            total_acc = 0
            dataset_test = get_dataset(1000, x_test, y_test, distorted_image_test_fn).make_one_shot_iterator()
            split = 10000 // 1000
            for _ in range(split):
                rotated_image_test, label_test = dataset_test.get_next()
                logits_test = model(rotated_image_test)
                prediction_test = tf.nn.softmax(logits_test)
                acc_test = model.accuracy(prediction_test, label_test).numpy()
                total_acc += acc_test
            print('test accuracy: {}'.format(total_acc / split))

        if i % 10 == 0:
            print("step: {}, loss: {}, train accuracy: {}".format(int(global_step), float(loss), float(acc)))


if __name__ == '__main__':
    main()

