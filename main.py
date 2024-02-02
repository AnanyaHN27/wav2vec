import math

import keras
import tensorflow as tf
from tensorflow.python.keras.models import Model
import keras.layers as layers


class FeatureEncoder(Model):
    def __init__(self, num_filters, kernel_size, strides):
        super(FeatureEncoder, self).__init__()
        self.filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.encoder_blocks = [self.encoder_block() for _ in range(5)]

    def layer_norm(self):
        # mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
        # gamma = tf.Variable(tf.ones_like(x))
        # beta = tf.Variable(tf.zeros_like(x))
        # x_norm = (x - mean) / tf.sqrt(var + epsilon)
        # return gamma * x_norm + beta
        return layers.LayerNormalization()

    def temporal_conv(self):
        """
        :param filters: Number of filters
        :param kernel_size: Size of the convolutional kernel
        :param strides: Stride of the convolution
        :param padding: Padding type
        :param activation: Activation function to apply
        :return: Convolutional layer
        """
        conv_layer = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='valid',
            activation=gelu
        )
        return conv_layer

    def encoder_block(self):
        return keras.Sequential([self.temporal_conv(), self.layer_norm()])

    def call(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / tf.constant(math.pi, dtype=x.dtype)) * (x + 0.044715 * tf.pow(x, 3))))


class ContextNetwork(keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(ContextNetwork, self).__init__()

        self.mha = keras.layers.MultiHeadAttention(
            key_dim=d_model // num_heads,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # In place of a positional embedding
        self.ffn = keras.Sequential([
            keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu'),
            keras.layers.LayerNormalization(),
            keras.layers.Conv1D(filters=d_model, kernel_size=1)
        ])

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        self.layerNorm1 = keras.layers.LayerNormalization()
        self.layerNorm2 = keras.layers.LayerNormalization()

        self.gelu = tf.keras.layers.Activation('gelu')

    def call(self, inputs, training=True):
        # Attention of the input
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layerNorm1(inputs + attn_output)

        # Output of convolution
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Adding the output of convolution to attention of input
        out2 = self.layerNorm2(out1 + ffn_output)

        return self.gelu(out2)
