import tensorflow as tf


class MixtureOfExpertsAdapt(tf.keras.Model):
    def __init__(
        self,
        experts,
        stock_temporal,
        macro_temporal,
        gate,
        embedding=None,
        use_stock_classification: bool = False,
        use_uncertainty: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Definitions
        self.experts = experts
        self.embedding = embedding
        self.use_stock_classification = use_stock_classification
        self.use_uncertainty = use_uncertainty

        # Modeling

        # Stock temporal
        self.stock_temporal = stock_temporal

        # Macro Temporal
        self.macro_temporal = macro_temporal

        # Experts
        self.experts = [expert for expert in experts]

        # Gating Mechanism
        self.gate = gate

    def call(self, inputs, training=False, return_gating=False, debug=False):
        sic_2 = inputs["sic_2"]
        stock_sequence = inputs["stock_sequence"]
        stock_classification = inputs["stock_classification"]
        macro_sequence = inputs["macro_sequence"]

        # Sequences

        # Stock Sequence
        stock_temporal_output = self.stock_temporal(stock_sequence)
        if debug:
            print("stock_temporal_output", stock_temporal_output)

        # Macro Sequence
        macro_temporal_output = self.macro_temporal(macro_sequence)
        if debug:
            print("macro_temporal_output", macro_temporal_output)

        # Concatenate output
        concatenated_stock_macro = tf.concat(
            [stock_temporal_output, macro_temporal_output], axis=2
        )
        if debug:
            print("concatenated_stock_macro", concatenated_stock_macro)

        # Experts

        # Pass the input into each expert, resulting in list of tensors each of shape (batch_size, output_dim) of length num_experts
        expert_outputs = [expert(concatenated_stock_macro) for expert in self.experts]

        # # Stack the outputs along a new axis (batch_size (2nd)), resulting in shape (batch_size, num_experts, output_dim)
        expert_outputs = tf.stack(expert_outputs, axis=1)

        # Remove the last dimension if output_dim == 1, resulting in shape (batch_size, num_experts)
        expert_outputs = tf.squeeze(expert_outputs, axis=-1)

        # Embedding

        # Embedding input shape = (batch_size, input_length)
        # Embedding output shape = (batch_size, input_length, output_dim)
        if self.embedding is not None:
            embed_outputs = self.embedding(sic_2)

        gating_input = embed_outputs

        # Gating Input
        if self.embedding is not None and self.use_stock_classification:
            print("1")
            gating_input = tf.concat([embed_outputs, stock_classification], axis=0)
        elif self.embedding is not None:
            print("2")
            gating_input = embed_outputs
        else:
            print("3")
            gating_input = stock_classification

        # Gating

        # Pass the input into the gating mechanism
        gating_weights = self.gate(gating_input)

        # Get the output by a weighted average of the experts output by the weights of the gating mechanism
        weighted_output = tf.reduce_sum(
            expert_outputs * gating_weights, axis=1, keepdims=True
        )

        if debug:
            return weighted_output, {
                "sic_2": sic_2,
                "stock_sequence": stock_sequence,
                "expert_outputs": expert_outputs,
                "gating_weights": gating_weights,
            }

        if return_gating:
            return weighted_output, gating_weights

        return weighted_output

    # def get_config(self):
    #     config = super().get_config()
    #     config.update(
    #         {
    #             "input_info": self.input_info,
    #             "experts_info": self.experts_info,
    #             "stock_temporal": self.stock_temporal,
    #             "macro_temporal": self.macro_temporal,
    #             "gate": self.gate,
    #             "use_emb": self.use_emb,
    #             "use_stock_classification": self.use_stock_classification,
    #             "use_uncertainty": self.use_uncertainty,
    #             "emb_output": self.emb_output,
    #         }
    #     )
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)
