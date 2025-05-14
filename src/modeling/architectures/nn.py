from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from src.modeling.layers.mc_dropout import MCDropout


def build_nn(
    input_shape,
    units: list,
    activations: list,
    use_layernorm: bool = False,
    mc_dropout: bool = False,
    dropout_rate: float = 0.0,
    output_units: int = 1,
    output_activation: str = "linear",
    optimizer="adam",
    loss="mse",
    model_name: str = None,
    show_summary: bool = False,
):
    """
    Builds a fully connected feedforward neural network model with optional dropout.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (features,).

    layers : int
        Number of Dense layers.

    units : list of int
        Number of units per Dense layer. Length must equal `layers`.

    activations : list of str
        Activation function for each Dense layer.

    dropout_rate : float
        Dropout rate after each Dense layer.

    output_units : int
        Number of output units.

    output_activation : str
        Activation function for the output layer.

    optimizer : str or keras.optimizers.Optimizer
        Optimizer for model compilation.

    loss : str or keras.losses.Loss
        Loss function for model compilation.

    model_name : str
        Optional name for the model.

    show_summary : bool
        If True, prints the model summary.

    Returns
    -------
    model : keras.Model
        Compiled model.
    """
    model = Sequential(name=model_name)
    model.add(Input(shape=input_shape))

    for i in range(len(units)):
        model.add(Dense(units[i], activation=activations[i], name=f"dense_{i}"))
        if use_layernorm:
            model.add(LayerNormalization())
        if dropout_rate > 0:
            if mc_dropout:
                model.add(MCDropout(dropout_rate, name=f"mc_dropout_{i}"))
            else:
                model.add(Dropout(dropout_rate, name=f"dropout_{i}"))

    model.add(Dense(output_units, activation=output_activation, name="output"))

    model.compile(optimizer=optimizer, loss=loss)

    if show_summary:
        model.summary()

    return model
