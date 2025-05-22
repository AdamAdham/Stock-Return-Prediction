from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Flatten,
)
from tensorflow.keras.optimizers import Adam

from src.modeling.layers.mc_dropout import MCDropout


def build_lstm(
    input_shape,
    hidden_layers: int,
    units: list,
    activations: list,
    return_sequences: bool,
    mc_dropout: bool = False,
    dropout_rate: float = 0.0,
    use_layernorm=False,
    regularizer=None,
    output_units: int = 1,
    output_activation: str = "linear",
    model_name: str = None,
    show_summary: bool = True,
):
    """
    Builds an LSTM model with optional attention and dropout.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (timesteps, features).

    layers : int
        Number of LSTM layers.

    units : list of int
        Number of units per LSTM layer. Length should equal `layers`.

    attention : bool
        If True, adds a simple attention placeholder after the last LSTM layer.

    dropout_rate : float
        Dropout rate to apply after each LSTM layer (0.0 means no dropout).

    output_units : int
        Number of units in the output layer.

    output_activation : str
        Activation function for the output layer (e.g., 'linear', 'sigmoid').

    show_summary : bool
        If True, prints the model summary.

    Returns
    -------
    model : keras.Model
        Compiled LSTM model.
    """
    if len(units) != hidden_layers:
        raise ValueError("Length of `units` must match the number of hidden_layers.")

    model = Sequential(name=model_name)
    model.add(Input(shape=input_shape))

    for i in range(hidden_layers):
        return_seq = True if i < hidden_layers - 1 else return_sequences
        model.add(
            LSTM(
                units[i],
                activations[i],
                return_sequences=return_seq,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
            )
        )
        if use_layernorm:
            model.add(LayerNormalization())
        if dropout_rate > 0:
            if mc_dropout:
                model.add(MCDropout(dropout_rate, name=f"mc_dropout_{i}"))
            else:
                model.add(Dropout(dropout_rate))

    if not return_sequences:
        model.add(
            Dense(
                output_units,
                activation=output_activation,
                kernel_regularizer=regularizer,
            )
        )

    model.compile(optimizer="adam", loss="mse")

    if show_summary:
        model.summary()

    return model


def build_lstm_attention(
    input_shape,
    num_blocks=2,
    units=64,
    activations="relu",
    num_heads=4,
    key_dim=None,
    attention_dropout=0,
    mc_dropout=False,
    dropout_rate=0.0,
    use_residual=False,
    use_layernorm=False,
    regularizer=None,
    use_pooling=True,
    pooling_layer=GlobalAveragePooling1D,
    return_sequences=False,
    output_units=1,
    output_activation="linear",
    optimizer=Adam,
    learning_rate=1e-3,
    loss="mse",
    model_name=None,
    show_summary=False,
):
    """
    Builds a model with repeated (LSTM → Attention) blocks.
    """

    # Make sure block-specific parameters are lists
    def ensure_list(x, name):
        if isinstance(x, list):
            return x
        print(f"{name} was not a list")
        return [x] * num_blocks

    units = ensure_list(units, "units")
    activations = ensure_list(activations, "activations")
    num_heads = ensure_list(num_heads, "num_heads")
    key_dim = ensure_list(key_dim, "key_dim") if key_dim is not None else units
    attention_dropout = ensure_list(attention_dropout, "attention_dropout")
    dropout_rate = ensure_list(dropout_rate, "dropout_rate")
    use_residual = ensure_list(use_residual, "use_residual")
    use_layernorm = ensure_list(use_layernorm, "use_layernorm")

    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        x_lstm = LSTM(
            units[i],
            return_sequences=True,
            activation=activations[i],
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer,
        )(x)

        x_attn = MultiHeadAttention(
            num_heads=num_heads[i],
            key_dim=key_dim[i],
            dropout=attention_dropout[i],
            kernel_regularizer=regularizer,
        )(x_lstm, x_lstm)

        if mc_dropout and dropout_rate[i] > 0.0:
            x_attn = MCDropout(dropout_rate[i])(x_attn)
        elif dropout_rate[i] > 0.0:
            x_attn = Dropout(dropout_rate[i])(x_attn)

        if use_residual[i]:
            x_attn = x_attn + x_lstm

        if use_layernorm[i]:
            x_attn = LayerNormalization()(x_attn)

        x = x_attn

    if return_sequences:
        outputs = x  # full sequence output
    else:
        if use_pooling:
            x = pooling_layer()(x)
        else:
            x = Flatten()(x)
        outputs = Dense(
            output_units, activation=output_activation, kernel_regularizer=regularizer
        )(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=loss)

    if show_summary:
        model.summary()
    return model
