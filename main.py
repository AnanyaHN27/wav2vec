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

class QuantizationModule(tf.keras.layers.Layer):
    def __init__(self, num_codebooks, num_entries_per_codebook, temperature=1.0):
        super(QuantizationModule).__init__()
        self.num_codebooks = num_codebooks
        self.num_entries_per_codebooks = num_entries_per_codebook
        self.temperature = temperature

    def gumbel_softmax(self, logits, temperature=1.0, epsilon=1e-20):
        # We set minval to epsilon so that we don't ever have to take the logarithm of 0
        gumbel_distributed_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), minval=epsilon, maxval=1.0)))
        gumbel_logits = (logits + gumbel_distributed_noise) / temperature
        softmax_probs = tf.nn.softmax(gumbel_logits, axis=-1)
        return softmax_probs

    def product_quantization(self, latent_representations, num_codebooks, num_entries_per_codebook):
        # Split the latent representations into subspaces for each codebook
        subspaces = tf.split(latent_representations, num_codebooks, axis=-1)

        quantized_subspaces = []
        for subspace in subspaces:
            # Choose one entry from each codebook
            codebook_entries = tf.split(subspace, num_entries_per_codebook, axis=-1)
            selected_entries = [tf.math.reduce_mean(entry, axis=-1, keepdims=True) for entry in codebook_entries]

            # Concatenate the selected entries
            quantized_subspace = tf.concat(selected_entries, axis=-1)
            quantized_subspaces.append(quantized_subspace)

        # Concatenate quantized subspaces to get the final quantized representation
        quantized_representation = tf.concat(quantized_subspaces, axis=-1)

        return quantized_representation

    def call(self, latent_representations, training=True):
        # Apply product quantization
        quantized_representation = self.product_quantization(latent_representations, self.num_codebooks,
                                                        self.num_entries_per_codebook)

        # Apply Gumbel softmax for differentiability during training
        gumbel_probs = self.gumbel_softmax(quantized_representation, temperature=self.temperature)

        return gumbel_probs

def create_model():
    T = 100  # number of time steps
    input_feature_dim = 64  # feature dimension
    num_codebooks = 4
    num_entries_per_codebook = 8
    temperature = 0.5

    raw_audio_input = tf.keras.Input(shape=(T, input_feature_dim))
    encoder_model = FeatureEncoder()
    context_model = ContextNetwork()
    quantization_model = QuantizationModule(num_codebooks=num_codebooks, num_entries_per_codebook=num_entries_per_codebook, temperature=temperature)

    latent_representations = encoder_model(raw_audio_input)
    contextualized_representations = context_model(latent_representations)
    discretized_representations = quantization_model(latent_representations)

    model_output = tf.keras.layers.Concatenate()([contextualized_representations, discretized_representations])

    overall_model = tf.keras.Model(inputs=raw_audio_input, outputs=model_output)
