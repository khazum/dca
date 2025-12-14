# FILE: dca/loss.py
from keras import ops
from keras.losses import Loss
import tensorflow as tf
from .layers import lgamma

def _nelem(x):
    # Safely calculate the number of non-NaN elements
    is_not_nan = ops.logical_not(ops.isnan(x))
    count = ops.sum(ops.cast(is_not_nan, "float32"))
    # Ensure the output dtype matches the input dtype for consistency
    return ops.cast(count, x.dtype)

def _reduce_mean(x):
    # Calculate mean while handling NaNs robustly
    nelem = _nelem(x)
    x = ops.nan_to_num(x)
    # Use divide_no_nan to prevent division by zero if nelem is 0
    return ops.divide_no_nan(ops.sum(x), nelem)

def mse_loss(y_true, y_pred):
    ret = ops.square(y_pred - y_true)
    return _reduce_mean(ret)

def _mean_over_genes(per_gene):
    """Average a per-gene loss tensor over genes, keeping per-sample shape."""
    per_gene = ops.nan_to_num(per_gene, nan=1e9, posinf=1e9, neginf=-1e9)
    gene_dim = ops.cast(ops.shape(per_gene)[-1], per_gene.dtype)
    return ops.sum(per_gene, axis=-1) / ops.maximum(gene_dim, ops.cast(1.0, per_gene.dtype))

def _nb_nll(y_true, mu, theta, eps):
    """Negative log-likelihood of NB(y | mu, theta) with stable log1p form."""
    mu = ops.maximum(mu, eps)
    theta = ops.maximum(theta, eps)
    t1 = lgamma(theta + eps) + lgamma(y_true + 1.0) - lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * ops.log1p(mu / (theta + eps)) + y_true * (ops.log(theta + eps) - ops.log(mu + eps))
    return t1 + t2

class WrappedLoss(Loss):
    def __init__(self, base_loss):
        # Use default reduction ("sum_over_batch_size").
        super().__init__(name="wrapped_dca_loss")
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        """Return batch-mean of per-sample, per-gene losses."""
        
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        
        try:
            # Attempt to use the base loss with mean=False if supported
            per_gene = self.base_loss(y_true, y_pred, mean=False)  # (B, G)
        except TypeError:
            # Fallback for built-in Keras losses (e.g., MSE)
            per_gene = ops.square(y_true - y_pred)  # (B, G)

        # Match legacy DCA reduction: mean over all elements (batch + genes)
        return _reduce_mean(per_gene)

class PackedNBLoss(Loss):
    def __init__(self, eps=1e-10):
        super().__init__(name="packed_nb_nll")
        self.eps = eps

    def call(self, y_true, y_pred):
        # y_pred is [mu, theta]
        if (y_pred.shape[-1] is not None) and (y_pred.shape[-1] % 2 != 0):
            raise ValueError(f"PackedNBLoss expects even last-dim, got {y_pred.shape[-1]}")
        mu, theta = ops.split(y_pred, 2, axis=-1)
        eps = ops.cast(self.eps, mu.dtype)

        # keep theta in a safe numeric range
        theta = ops.clip(theta, eps, ops.cast(1e6, theta.dtype))
        per_gene = _nb_nll(y_true, mu, theta, eps)
        return _reduce_mean(per_gene)

class PackedZINBLoss(Loss):
    def __init__(self, ridge_lambda=0.0, eps=1e-10):
        super().__init__(name="packed_zinb_nll")
        self.ridge_lambda = ridge_lambda
        self.eps = eps

    def call(self, y_true, y_pred):
        # y_pred is [mu, theta, pi]
        if (y_pred.shape[-1] is None) or (y_pred.shape[-1] % 3 != 0):
            raise ValueError(f"PackedZINBLoss expects last-dim to be 3*genes, got {y_pred.shape[-1]}")
        
        mu, theta, pi = ops.split(y_pred, 3, axis=-1)
        eps = ops.cast(self.eps, mu.dtype)
        
        # Clamp parameters for stability
        mu = ops.maximum(mu, eps)
        theta = ops.clip(theta, eps, ops.cast(1e6, theta.dtype))
        one = ops.cast(1.0, pi.dtype)
        pi = ops.clip(pi, eps, one - eps)

        # --- NB NLL (non-zero case) ---
        nb_nll = _nb_nll(y_true, mu, theta, eps)
        
        # Combine with (1-pi)
        nb_case = nb_nll - ops.log(1.0 - pi + eps)
        
        # --- ZI part (zero case) ---
        # P_NB(0) = (theta / (theta + mu))^theta
        zero_nb = ops.power(theta / (theta + mu + eps), theta)

        # P(Y=0) = pi + (1-pi)*P_NB(0)
        zero_case = -ops.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
        # Select case based on whether y_true is zero
        result = ops.where(ops.less(y_true, ops.cast(1e-8, y_true.dtype)), zero_case, nb_case)
        
        # Regularization (Ridge penalty on pi) — use sum over pi to match legacy strength
        per_sample = _mean_over_genes(result)
        if self.ridge_lambda > 0:
            ridge_vec = ops.cast(self.ridge_lambda, per_sample.dtype) * ops.sum(ops.square(pi), axis=-1)
            per_sample = per_sample + ridge_vec
        return _reduce_mean(per_sample)

def poisson_loss(y_true, y_pred, mean=True):
    y_pred = ops.cast(y_pred, "float32")
    y_true = ops.cast(y_true, "float32")

    y_true = ops.nan_to_num(y_true)
    # Poisson NLL: mu - y*log(mu) + log(y!)
    ret = y_pred - y_true * ops.log(y_pred + 1e-10) + lgamma(y_true + 1.0)
    if mean:
        nelem = _nelem(y_true)
        return ops.divide_no_nan(ops.sum(ret), nelem)
    else:
        return ret

# TensorFlow-native packed losses (avoid Keras-ops reduction pitfalls)
def packed_nb_loss_tf(y_true, y_pred, eps=1e-10):
    mu, theta = tf.split(y_pred, 2, axis=-1)
    mu = tf.maximum(mu, eps)
    theta = tf.clip_by_value(theta, eps, 1e6)
    # per-gene
    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.math.log1p(mu / (theta + eps)) + y_true * (tf.math.log(theta + eps) - tf.math.log(mu + eps))
    per_gene = t1 + t2  # (B, G)
    per_sample = tf.reduce_mean(per_gene, axis=-1)  # mean over genes
    return tf.reduce_mean(per_sample)  # mean over batch

def packed_zinb_loss_tf(y_true, y_pred, ridge_lambda=0.0, eps=1e-10):
    mu, theta, pi = tf.split(y_pred, 3, axis=-1)
    mu = tf.maximum(mu, eps)
    theta = tf.clip_by_value(theta, eps, 1e6)
    pi = tf.clip_by_value(pi, eps, 1.0 - eps)

    nb_t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
    nb_t2 = (theta + y_true) * tf.math.log1p(mu / (theta + eps)) + y_true * (tf.math.log(theta + eps) - tf.math.log(mu + eps))
    nb_nll = nb_t1 + nb_t2
    nb_case = nb_nll - tf.math.log(1.0 - pi + eps)

    zero_nb = tf.pow(theta / (theta + mu + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)

    result = tf.where(tf.less(y_true, tf.cast(1e-8, y_true.dtype)), zero_case, nb_case)
    per_sample = tf.reduce_mean(result, axis=-1)
    if ridge_lambda > 0:
        per_sample = per_sample + tf.cast(ridge_lambda, result.dtype) * tf.reduce_sum(tf.square(pi), axis=-1)
    return tf.reduce_mean(per_sample)
