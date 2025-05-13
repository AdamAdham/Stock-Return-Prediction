from tensorflow.keras.layers import Dropout


class MCDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
