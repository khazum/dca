# Copyright 2016 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random

from . import io
from .network import AE_types

import numpy as np
import scipy.sparse as sp

import tensorflow as tf
from tensorflow.keras import optimizers as opt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.losses import Loss
from tensorflow.keras import ops

class WrappedLoss(Loss):
    def __init__(self, base_loss):
        # Use "sum" so Keras doesn't apply an extra averaging step.
        super().__init__(reduction="sum", name="wrapped_dca_loss")
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        """Return batch-mean of per-sample, per-gene losses.
        Matches TF1 impl which averaged over *all* elements (B*G)."""
        try:
            per_gene = self.base_loss(y_true, y_pred, mean=False)  # (B, G)
        except TypeError:
            # Fallback for built-in Keras losses without 'mean' arg (e.g., MSE)
            per_gene = tf.math.squared_difference(y_true, y_pred)  # (B, G)
        g = tf.cast(tf.shape(per_gene)[-1], per_gene.dtype)
        per_sample = tf.reduce_sum(per_gene, axis=-1) / tf.maximum(g, 1.0)  # (B,)
        return tf.reduce_mean(per_sample)  # scalar

class PackedNBLoss(Loss):
    def __init__(self, eps=1e-10):
        super().__init__(reduction="sum", name="packed_nb_nll")
        self.eps = eps

    def call(self, y_true, y_pred):
        # Ensure last dimension is even so we can split into [mu|theta]
        last = ops.shape(y_pred)[-1]
        if (y_pred.shape[-1] is not None) and (y_pred.shape[-1] % 2 != 0):
            raise ValueError(f"PackedNBLoss expects even last-dim, got {y_pred.shape[-1]}")
        mu, theta = ops.split(y_pred, 2, axis=-1)

        eps = tf.cast(self.eps, mu.dtype)

        # Negative log-likelihood of NB (pure TF ops)
        t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * tf.math.log1p(mu / (theta + eps)) + y_true * (tf.math.log(theta + eps) - tf.math.log(mu + eps))
        per_gene = t1 + t2                                    # (B, G)
        g = tf.cast(tf.shape(per_gene)[-1], per_gene.dtype)
        per_sample = tf.reduce_sum(per_gene, axis=-1) / tf.maximum(g, 1.0)  # (B,)
        return tf.reduce_mean(per_sample)                     # scalar
    
def train(
    adata,
    network,
    output_dir=None,
    optimizer="RMSprop",
    learning_rate=None,
    epochs=300,
    reduce_lr=10,
    output_subset=None,
    use_raw_as_output=True,
    early_stop=15,
    batch_size=32,
    clip_grad=5.0,
    save_weights=False,
    validation_split=0.1,
    tensorboard=False,
    verbose=True,
    threads=None,
    **kwds
):
    if threads:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(threads))
            tf.config.threading.set_inter_op_parallelism_threads(int(threads))
        except Exception:
            pass
        
    model = network.model
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Build optimizer from its name (e.g., "RMSprop") using tf.keras
    opt_cls = getattr(opt, optimizer)
    optimizer = (
        opt_cls(clipvalue=clip_grad)
        if learning_rate is None
        else opt_cls(learning_rate=learning_rate, clipvalue=clip_grad)
    )

    if any(l.name == "pack" for l in model.layers):
        loss_fn = PackedNBLoss()
    else:
        loss_fn = WrappedLoss(network.loss)

    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=True, jit_compile=False)

    # Callbacks
    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(
            filepath="%s/weights.hdf5" % output_dir,
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
        )
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(
            monitor="val_loss", patience=reduce_lr, verbose=verbose
        )
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor="val_loss", patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, "tb")
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
        callbacks.append(tb_cb)

    if verbose:
        model.summary()

    sf = np.asarray(adata.obs.size_factors).reshape(-1, 1)  # (n, 1)
    y = adata.raw.X if use_raw_as_output else adata.X
    if sp.issparse(y):
        y = y.A  # dense
        
    # Keras expects dense arrays
    X = adata.X.A if sp.issparse(adata.X) else np.asarray(adata.X)
    inputs = {"count": X, "size_factors": sf}
    output = y  # already dense
    if output_subset:
        gene_idx = [np.where(adata.raw.var_names == x)[0][0] for x in output_subset]
        # slice the dense array instead of reloading a sparse matrix
        output = output[:, gene_idx]
    # ensure dtype
    output = np.asarray(output, dtype=np.float32)

    history = model.fit(
        inputs,
        output,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_split=validation_split,
        verbose=verbose,
        **kwds
    )

    return history


def train_with_args(args):
    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ["PYTHONHASHSEED"] = "0"

    if args.threads:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(args.threads))
            tf.config.threading.set_inter_op_parallelism_threads(int(args.threads))
        except Exception:
            pass

    adata = io.read_dataset(
        args.input,
        transpose=(not args.transpose),  # assume gene x cell by default
        check_counts=args.checkcounts,
        test_split=args.testsplit,
    )

    adata = io.normalize(
        adata,
        size_factors=args.sizefactors,
        logtrans_input=args.loginput,
        normalize_input=args.norminput,
    )

    if args.denoisesubset:
        genelist = list(set(io.read_genelist(args.denoisesubset)))
        assert (
            len(set(genelist) - set(adata.var_names.values)) == 0
        ), "Gene list is not overlapping with genes from the dataset"
        output_size = len(genelist)
    else:
        genelist = None
        output_size = adata.n_vars

    hidden_size = [int(x) for x in args.hiddensize.split(",")]
    hidden_dropout = [float(x) for x in args.dropoutrate.split(",")]
    if len(hidden_dropout) == 1:
        hidden_dropout = hidden_dropout[0]

    assert args.type in AE_types, "loss type not supported"
    input_size = adata.n_vars

    net = AE_types[args.type](
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        l2_coef=args.l2,
        l1_coef=args.l1,
        l2_enc_coef=args.l2enc,
        l1_enc_coef=args.l1enc,
        ridge=args.ridge,
        hidden_dropout=hidden_dropout,
        input_dropout=args.inputdropout,
        batchnorm=args.batchnorm,
        activation=args.activation,
        init=args.init,
        debug=args.debug,
        file_path=args.outputdir,
    )

    net.save()
    net.build()

    losses = train(
        adata[adata.obs.dca_split == "train"],
        net,
        output_dir=args.outputdir,
        learning_rate=args.learningrate,
        epochs=args.epochs,
        batch_size=args.batchsize,
        early_stop=args.earlystop,
        reduce_lr=args.reducelr,
        output_subset=genelist,
        optimizer=args.optimizer,
        clip_grad=args.gradclip,
        save_weights=args.saveweights,
        tensorboard=args.tensorboard,
    )

    if genelist:
        predict_columns = adata.var_names[
            [np.where(adata.var_names == x)[0][0] for x in genelist]
        ]
    else:
        predict_columns = adata.var_names

    net.predict(adata, mode="full", return_info=True)
    net.write(adata, args.outputdir, mode="full", colnames=predict_columns)
