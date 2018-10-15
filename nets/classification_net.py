"""Classification Net
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from nets.deformable_conv_layer import DeformableConvLayer


class ClassificationNet(keras.Model):
    def __init__(self, num_class, **kwargs):
        super().__init__(self, **kwargs)
        # classification net
        self.conv1 = DeformableConvLayer(32, [5, 5], num_deformable_group=1, activation='relu')  # out 24
        # self.conv1 = Conv2D(32, [5, 5], activation='relu')
        self.conv2 = Conv2D(32, [5, 5], activation='relu')  # out 20
        self.max_pool1 = MaxPool2D(2, [2, 2])  # out 10
        self.conv3 = Conv2D(32, [5, 5], activation='relu')  # out 6
        self.conv4 = Conv2D(32, [5, 5], activation='relu')  # out 2
        self.max_pool2 = MaxPool2D(2, [2, 2])  # out 1
        self.flatten = Flatten()
        self.fc = Dense(num_class)

    def call(self, inputs, training=None, mask=None):
        net = self.conv1(inputs)
        net = self.conv2(net)
        net = self.max_pool1(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.max_pool2(net)
        net = self.flatten(net)
        logits = self.fc(net)
        return logits

    def train(self, optimizer, x, y):
        with tf.GradientTape() as tape:
            logits = self.__call__(x)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, self.variables)
        optimizer.apply_gradients(zip(grads, self.variables))
        return loss, tf.nn.softmax(logits)

    def accuracy(self, prediction, y):
        eq = tf.to_float(tf.equal(tf.argmax(prediction, axis=-1), tf.argmax(y, axis=-1)))
        return tf.reduce_mean(eq)
