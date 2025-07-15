from src.modeling.architectures.lstm import build_lstm, build_lstm_attention
from src.modeling.architectures.transformer import build_transformer
from src.modeling.architectures.nn import build_nn


def model_builder(input_shape, model_type, **kwargs):
    if model_type == "lstm":
        return build_lstm(input_shape=input_shape, **kwargs)
    elif model_type == "lstm_attention":
        return build_lstm_attention(input_shape=input_shape, **kwargs)
    elif model_type == "transformer":
        return build_transformer(input_shape=input_shape, **kwargs)
    elif model_type == "dense":
        return build_nn(input_shape=input_shape, **kwargs)
