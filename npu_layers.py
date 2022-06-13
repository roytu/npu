
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Zeros, Constant, GlorotUniform

class NauLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="Nau", **kwargs):
        super(NauLayer, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.A = self.add_weight(
                "A",
                shape=(input_shape[-1], self.num_outputs),
                initializer=GlorotUniform(),
                trainable=True)

    def call(self, inputs, training=None):
        out = tf.matmul(inputs, self.A)

        # Add regularized losses
        def get_reg_loss(A):
            loss = tf.reduce_sum(tf.math.minimum(tf.abs(A), tf.abs(1 - tf.abs(A))))
            return loss

        reg_loss = get_reg_loss(self.A)
        self.add_loss(reg_loss)
        self.add_metric(reg_loss, name="A_loss")

        return out

class NpuLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name="Npu", **kwargs):
        super(NpuLayer, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.W_r = self.add_weight(
                "W_r",
                shape=(input_shape[-1], self.num_outputs),
                initializer=GlorotUniform(),
                trainable=True)

        self.W_i = self.add_weight(
                "W_i",
                shape=(input_shape[-1], self.num_outputs),
                initializer=Zeros(),
                trainable=True)

        self.g = self.add_weight(
                "g",
                shape=(self.num_outputs,),
                initializer=Constant(0.5),
                trainable=True)

    def call(self, inputs, training=None):
        eps = 1e-7

        # Calculate intermediate values
        r = self.g * tf.abs(inputs) + (1. - self.g)
        k = tf.where(tf.less(inputs, 0.), self.g, 0.)

        logr = tf.math.log(tf.clip_by_value(r, eps, 1e10))

        y1 = tf.exp(tf.matmul(logr, self.W_r) - np.pi * tf.matmul(k, self.W_i))
        y2 = tf.cos(tf.matmul(logr, self.W_i) + np.pi * tf.matmul(k, self.W_r))

        out = y1 * y2

        # Add regularized losses
        def get_W_reg_loss(x):
            loss = tf.reduce_sum(tf.math.minimum(tf.abs(x), tf.abs(1 - x)))
            return loss

        def get_g_reg_loss(x):
            loss = tf.reduce_sum(tf.math.minimum(tf.abs(x), tf.abs(1 - x)))
            return loss

        Wr_loss = get_W_reg_loss(self.W_r)
        self.add_loss(Wr_loss)
        self.add_metric(Wr_loss, name="Wr_loss")

        Wi_loss = get_W_reg_loss(self.W_i)
        self.add_loss(Wi_loss)
        self.add_metric(Wi_loss, name="Wi_loss")

        g_loss = get_g_reg_loss(self.g)
        self.add_loss(g_loss)
        self.add_metric(g_loss, name="g_loss")

        return out

