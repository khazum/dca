# dca_patched/dca/layers.py
from tensorflow.keras.layers import Layer, Lambda, Dense, InputSpec
from tensorflow.keras import ops

class ConstantDispersionLayer(Layer):
    """
    Identity layer that carries a trainable dispersion parameter.
    """
    def build(self, input_shape):
        self.theta = self.add_weight(
            shape=(1, input_shape[1]),
            initializer="zeros",
            trainable=True,
            name="theta",
        )
        super().build(input_shape)

    def call(self, x):
        # Identity pass-through; dispersion is exposed via trainable weights.
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape

    @property
    def theta_exp(self):
        # Compute on-the-fly so it always reflects current weights.
        return ops.clip(ops.exp(self.theta), 1e-3, 1e4)

class SliceLayer(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("Input should be a list")
        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list), "SliceLayer input is not a list"
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class ElementwiseDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert (input_dim == self.units) or (self.units == 1), \
               "Input and output dims are not compatible"
        self.kernel = self.add_weight(
            shape=(self.units,),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        out = inputs * self.kernel  # broadcasting
        if self.use_bias:
            out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


# Keras-ops lambdas (no tf.*)
nan2zeroLayer = Lambda(lambda x: ops.nan_to_num(x))
ColwiseMultLayer = Lambda(lambda l: ops.multiply(l[0], ops.reshape(l[1], (-1, 1))))
