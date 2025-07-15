import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Flatten,
)
import numpy as np

from src.modeling.layers.mc_dropout import MCDropout


def get_sinusoidal_encoding(seq_len, d_model):
    """
    Generates a sinusoidal positional encoding matrix for input sequences.

    Positional encoding is used in transformer models to inject information about
    the order of the sequence into the input embeddings, since transformers
    do not have built-in recurrence or convolution.

    The encoding uses sine functions for even dimensions and cosine functions
    for odd dimensions, following the original Transformer paper:
    "Attention is All You Need" (Vaswani et al., 2017).

    Parameters
    ----------
    seq_len : int
        Length of the input sequence (number of time steps).
    d_model : int
        Dimensionality of the model/embedding vector.

    Returns
    -------
    tf.Tensor
        A tensor of shape (seq_len, d_model) containing the sinusoidal positional encodings.
    """

    # [:, np.newaxis] reshapes it into a 2D column vector of shape (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # (1, d_model)

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = position * angle_rates  # (seq_len, d_model)

    # Apply sin to even indices (0, 2, 4, ...)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices (1, 3, 5, ...)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.constant(angle_rads, dtype=tf.float32)


def build_transformer(
    input_shape,
    num_heads=4,
    num_blocks=1,
    model_dim=512,
    dense_interm_dim=1024,
    use_layernorm=True,
    use_residual=True,
    mc_dropout: bool = False,
    dropout_rate=0.0,
    regularizer=None,
    use_pooling=True,
    pooling_layer=GlobalAveragePooling1D,
    positional_encoding=True,
    sinusoidal_encoding: bool = True,
    output_dim=1,
    return_sequences=False,
    optimizer="adam",
    loss="mse",
    show_summary=False,
):
    inputs = tf.keras.Input(shape=input_shape)  # shape = (timesteps, features)
    x = inputs

    x = Dense(model_dim, kernel_regularizer=regularizer)(x)

    # Optional positional encoding
    if positional_encoding:
        # Creates a 1D tensor of position indices: e.g., [0, 1, ..., 11] if input_shape=(12, 19)
        if sinusoidal_encoding:
            pos_embedding = get_sinusoidal_encoding(input_shape[0], d_model=model_dim)
            pos_embedding = tf.expand_dims(
                pos_embedding, axis=0
            )  # shape: (1, timesteps, d_model)
        else:
            pos = tf.range(start=0, limit=input_shape[0], delta=1)

            # Learnable embedding: each position gets mapped to a vector of size model_dim (e.g., 19)
            pos_embedding = Embedding(input_dim=input_shape[0], output_dim=model_dim)(
                pos
            )  # shape: (12, 19)
            # Expand dims for broadcasting consistency across batches (optional, but clearer)
            pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # shape: (1, 12, 19)

        print(pos_embedding)
        x = x + pos_embedding

    prev_output = x
    # Transformer blocks
    for _ in range(num_blocks):
        key_dim = model_dim // num_heads
        output = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, kernel_regularizer=regularizer
        )(x, x)

        if use_residual:
            output = output + prev_output

        if use_layernorm:
            output = LayerNormalization()(output)

        if dropout_rate > 0:
            if mc_dropout:
                output = MCDropout(dropout_rate)(output)
            else:
                output = Dropout(dropout_rate)(output)

        ffn_output = Dense(dense_interm_dim)(output)
        ffn_output = Dense(model_dim)(ffn_output)

        if use_residual:
            output = output + ffn_output
        else:
            output = ffn_output

        if use_layernorm:
            output = LayerNormalization()(output)

        if dropout_rate > 0:
            if mc_dropout:
                output = MCDropout(dropout_rate)(output)
            else:
                output = Dropout(dropout_rate)(output)

        prev_output = output

    if not return_sequences:
        if use_pooling:
            output = pooling_layer()(output)
        else:
            output = Flatten()(output)

        output = Dense(output_dim, activation="linear", kernel_regularizer=regularizer)(
            output
        )

    model = Model(inputs, output, name="transformer_forecaster")
    model.compile(optimizer=optimizer, loss=loss)

    if show_summary:
        model.summary()

    return model
