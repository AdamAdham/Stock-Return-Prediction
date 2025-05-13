import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
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
    ff_dim=128,
    num_transformer_blocks=2,
    mc_dropout: bool = False,
    dropout_rate=0.0,
    positional_encoding=True,
    sinusoidal_encoding: bool = True,
    output_dim=1,
    return_sequences=False,
    optimizer="adam",
    loss="mse",
):
    inputs = tf.keras.Input(shape=input_shape)  # shape = (timesteps, features)
    x = inputs
    print("x")
    print(x)
    print("-" * 100)

    # Optional positional encoding
    if positional_encoding:
        # Creates a 1D tensor of position indices: e.g., [0, 1, ..., 11] if input_shape=(12, 19)
        if sinusoidal_encoding:
            pos_embedding = get_sinusoidal_encoding(
                input_shape[0], d_model=input_shape[1]
            )
            pos_embedding = tf.expand_dims(
                pos_embedding, axis=0
            )  # shape: (1, timesteps, d_model)
        else:
            pos = tf.range(start=0, limit=input_shape[0], delta=1)

            # Learnable embedding: each position gets mapped to a vector of size input_shape[1] (e.g., 19)
            pos_embedding = Embedding(input_dim=1000, output_dim=input_shape[1])(
                pos
            )  # shape: (12, 19)
            # Expand dims for broadcasting consistency across batches (optional, but clearer)
            pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # shape: (1, 12, 19)

        print(pos_embedding)
        x = x + pos_embedding

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(
            x, x
        )
        attn_output = Dropout(dropout_rate)(attn_output)
        x = LayerNormalization()(x + attn_output)

        ffn = Dense(ff_dim, activation="relu")(x)
        ffn = Dense(input_shape[1])(ffn)
        if dropout_rate > 0:
            if mc_dropout:
                ffn = MCDropout(dropout_rate)(ffn)
            else:
                ffn = Dropout(dropout_rate)(ffn)
        x = LayerNormalization()(x + ffn)

    if not return_sequences:
        x = GlobalAveragePooling1D()(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs, name="transformer_forecaster")
    model.compile(optimizer=optimizer, loss=loss)

    return model
