import numpy as np
from keras import ops
from keras.losses import Loss
from .layers import lgamma

def _nelem(x):
    is_not_nan = ops.logical_not(ops.isnan(x))
    count = ops.sum(ops.cast(is_not_nan, "float32"))
    return ops.cast(count, x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = ops.nan_to_num(x)
    return ops.divide_no_nan(ops.sum(x), nelem)

def mse_loss(y_true, y_pred):
    ret = ops.square(y_pred - y_true)

    return _reduce_mean(ret)

# --- Modern Packed Loss Implementations (Moved from train.py) ---

class WrappedLoss(Loss):
    def __init__(self, base_loss):
        # Use "sum" so Keras doesn't apply an extra averaging step. We handle averaging manually.
        super().__init__(reduction="sum", name="wrapped_dca_loss")
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        """Return batch-mean of per-sample, per-gene losses."""
        
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        
        try:
            per_gene = self.base_loss(y_true, y_pred, mean=False)  # (B, G)
        except TypeError:
            # Fallback for built-in Keras losses (e.g., MSE)
            per_gene = ops.square(y_true - y_pred)  # (B, G)

        # Stability: Use a large finite number instead of np.inf
        per_gene = ops.nan_to_num(per_gene, nan=1e9, posinf=1e9, neginf=-1e9)
        
        # Average over genes (G), then over batch (B)
        g = ops.cast(ops.shape(per_gene)[-1], per_gene.dtype)
        per_sample = ops.sum(per_gene, axis=-1) / ops.maximum(g, ops.cast(1.0, per_gene.dtype))
        return ops.mean(per_sample)

class PackedNBLoss(Loss):
    def __init__(self, eps=1e-10):
        super().__init__(reduction="sum", name="packed_nb_nll")
        self.eps = eps
        self.lgamma = lgamma

    def call(self, y_true, y_pred):
        # y_pred is [mu, theta]
        if (y_pred.shape[-1] is not None) and (y_pred.shape[-1] % 2 != 0):
            raise ValueError(f"PackedNBLoss expects even last-dim, got {y_pred.shape[-1]}")
        mu, theta = ops.split(y_pred, 2, axis=-1)
        eps = ops.cast(self.eps, mu.dtype)

        t1 = self.lgamma(theta + eps) + self.lgamma(y_true + 1.0) - self.lgamma(y_true + theta + eps)
        # Use log1p for better stability when mu/theta is small
        t2 = (theta + y_true) * ops.log1p(mu / (theta + eps)) + y_true * (ops.log(theta + eps) - ops.log(mu + eps))
        per_gene = t1 + t2

        # Stability: Use a large finite number instead of np.inf
        per_gene = ops.nan_to_num(per_gene, nan=1e9, posinf=1e9, neginf=-1e9)

        # Aggregation
        g = ops.cast(ops.shape(per_gene)[-1], per_gene.dtype)
        per_sample = ops.sum(per_gene, axis=-1) / ops.maximum(g, ops.cast(1.0, per_gene.dtype))
        return ops.mean(per_sample)

class PackedZINBLoss(Loss):
    def __init__(self, ridge_lambda=0.0, eps=1e-10):
        super().__init__(reduction="sum", name="packed_zinb_nll")
        self.ridge_lambda = ridge_lambda
        self.eps = eps
        self.lgamma = lgamma

    def call(self, y_true, y_pred):
        # y_pred is [mu, theta, pi]
        if (y_pred.shape[-1] is None) or (y_pred.shape[-1] % 3 != 0):
            raise ValueError(f"PackedZINBLoss expects last-dim to be 3*genes, got {y_pred.shape[-1]}")
        
        mu, theta, pi = ops.split(y_pred, 3, axis=-1)
        eps = ops.cast(self.eps, mu.dtype)
        
        # --- NB NLL (non-zero case) ---
        theta = ops.minimum(theta, ops.cast(1e6, theta.dtype))
        t1 = self.lgamma(theta + eps) + self.lgamma(y_true + 1.0) - self.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * ops.log(1.0 + (mu / (theta + eps))) + (y_true * (ops.log(theta + eps) - ops.log(mu + eps)))
        nb_nll = t1 + t2
        nb_case = nb_nll - ops.log(1.0 - pi + eps)
        
        # --- ZI part (zero case) ---
        zero_nb = ops.power(theta / (theta + mu + eps), theta)
        zero_case = -ops.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
        result = ops.where(ops.less(y_true, ops.cast(1e-8, y_true.dtype)), zero_case, nb_case)
        
        # Regularization
        ridge = ops.cast(self.ridge_lambda, result.dtype) * ops.square(pi)
        result += ridge

        # Stability: Use a large finite number instead of np.inf to avoid NaN gradients.
        result = ops.nan_to_num(result, nan=1e9, posinf=1e9, neginf=-1e9)
        
        # --- Aggregation ---
        g = ops.cast(ops.shape(result)[-1], result.dtype)
        per_sample = ops.sum(result, axis=-1) / ops.maximum(g, ops.cast(1.0, result.dtype))
        return ops.mean(per_sample)


# In the implementations, I try to keep the function signature
# similar to those of Keras objective functions so that
# later on we can use them in Keras smoothly:
# https://github.com/fchollet/keras/blob/master/keras/objectives.py#L7
def poisson_loss(y_true, y_pred, mean=True):
    y_pred = ops.cast(y_pred, "float32")
    y_true = ops.cast(y_true, "float32")

    y_true = ops.nan_to_num(y_true)
    # elementwise Poisson NLL
    ret = y_pred - y_true * ops.log(y_pred + 1e-10) + lgamma(y_true + 1.0)
    if mean:
        nelem = _nelem(y_true)
        return ops.divide_no_nan(ops.sum(ret), nelem)
    else:
        return ret


# We need a class (or closure) here,
# because it's not possible to
# pass extra arguments to Keras loss functions
# See https://github.com/fchollet/keras/issues/2121

# dispersion (theta) parameter is a scalar by default.
# scale_factor scales the nbinom mean before the
# calculation of the loss to balance the
# learning rates of theta and network weights
class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                    scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta
        self.lgamma = lgamma

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        # Keep everything in Keras ops (no tf.* scope)
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32") * scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = ops.nan_to_num(y_true)

        # Clip theta
        # Use Keras ops to keep compatibility with KerasTensors.
        theta = ops.minimum(self.theta, ops.cast(1e6, self.theta.dtype))

        t1 = self.lgamma(theta + eps) + self.lgamma(y_true + 1.0) - self.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * ops.log(1.0 + (y_pred / (theta + eps))) + (y_true * (ops.log(theta + eps) - ops.log(y_pred + eps)))


        final = t1 + t2

        final = ops.nan_to_num(final, nan=np.inf, posinf=None, neginf=None)

        if mean:
            if self.masking:
                final = ops.divide(ops.sum(final), nelem)
            else:
                final = ops.mean(final)

        return final

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        
        nb_case = super().loss(y_true, y_pred, mean=False) - ops.log(1.0 - self.pi + eps)

        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32") * scale_factor
        theta = ops.minimum(self.theta, ops.cast(1e6, self.theta.dtype))

        zero_nb = ops.power(theta / (theta + y_pred + eps), theta)
        zero_case = -ops.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = ops.where(ops.less(y_true, ops.cast(1e-8, y_true.dtype)), zero_case, nb_case)
        ridge = self.ridge_lambda * ops.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = ops.mean(result)

        result = ops.nan_to_num(result, nan=np.inf, posinf=None, neginf=None)

        # if self.debug:
        #     tf.summary.histogram('nb_case', nb_case)
        #     tf.summary.histogram('zero_nb', zero_nb)
        #     tf.summary.histogram('zero_case', zero_case)
        #     tf.summary.histogram('ridge', ridge)

        return result
