# dca_patched/dca/layers.py
from keras.layers import Layer, Lambda, Dense, InputSpec
from keras import ops

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
        self.ew_kernel = self.add_weight(
            shape=(self.units,),
            initializer=self.kernel_initializer,
            name="ew_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.ew_bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="ew_bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.ew_bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        out = inputs * self.ew_kernel  # broadcasting
        if self.use_bias:
            out = out + self.ew_bias
        if self.activation is not None:
            out = self.activation(out)
        return out

_LANCZOS = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]
_LOG_SQRT_2PI = 0.9189385332046727  # 0.5*log(2π)

def lgamma(x):
    # backend-agnostic (works on KerasTensors); no tf.* calls
    x = ops.cast(x, "float32")
    x = ops.maximum(x, ops.cast(1e-7, x.dtype))  # avoid poles
    z = x - 1.0

    a = ops.cast(_LANCZOS[0], x.dtype)
    # sum_{k=1..8} c_k / (z + k)
    for k, c in enumerate(_LANCZOS[1:], start=1):
        a = a + ops.cast(c, x.dtype) / (z + ops.cast(k, x.dtype))

    t = z + ops.cast(7.0 + 0.5, x.dtype)  # g + 0.5
    return (_LOG_SQRT_2PI
            + ops.log(a)
            + (z + 0.5) * ops.log(t)
            - t)

ColwiseMultLayer = Lambda(lambda l: ops.multiply(l[0], ops.reshape(l[1], (-1, 1))))
