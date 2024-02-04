import math
import tensorflow as tf

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / tf.constant(math.pi, dtype=x.dtype)) * (x + 0.044715 * tf.pow(x, 3))))

class FeatureEncoder(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, strides):
        super(FeatureEncoder, self).__init__()
        self.filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.encoder_blocks = [self.encoder_block() for _ in range(5)]

    def layer_norm(self):
        return tf.keras.layers.LayerNormalization()

    def temporal_conv(self):
        conv_layer = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='valid',
            activation=gelu
        )
        return conv_layer

    def encoder_block(self):
        return tf.keras.Sequential([self.temporal_conv(), self.layer_norm()])

    def call(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        return x


class ContextNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1):
        super(ContextNetwork, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model // num_heads,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # Instead of a positional embedding
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        ])

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layerNorm1 = tf.keras.layers.LayerNormalization()
        self.layerNorm2 = tf.keras.layers.LayerNormalization()

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

# Example Usage:
# Assuming input_data is your input data
input_data = tf.random.normal(())#batch_size=3, sequence_length=120, input_channels))

# Creating instances of FeatureEncoder and ContextNetwork
feature_encoder = FeatureEncoder(num_filters=64, kernel_size=3, strides=1)
context_network = ContextNetwork(d_model=256, num_heads=4, ff_dim=512, dropout_rate=0.1)

# Forward pass through the FeatureEncoder
latent_speech_representations = feature_encoder(input_data)

# Forward pass through the ContextNetwork
contextualized_representations = context_network(latent_speech_representations)
