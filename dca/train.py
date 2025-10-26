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
from .loss import WrappedLoss, PackedNBLoss, PackedZINBLoss

import numpy as np
import scipy.sparse as sp

import keras
from keras import optimizers as opt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import gc

     
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
    # if threads:
    #     try:
    #         tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    #         tf.config.threading.set_inter_op_parallelism_threads(int(threads))
    #     except Exception:
    #         pass
        
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

    is_packed = any(l.name == "pack" for l in model.layers)
    if not is_packed:
        # Fallback to old method if 'pack' layer isn't found
        loss_fn = WrappedLoss(network.loss)
    else:
        # 'pack' layer exists, check output shape to decide loss
        # NB models pack [mu, theta] -> 2 * G
        # ZINB models pack [mu, theta, pi] -> 3 * G
        output_dim = model.output_shape[-1]
        # Use network.output_size, as input_size might differ
        genes_dim = network.output_size 
        
        if output_dim == 2 * genes_dim:
            loss_fn = PackedNBLoss()
        elif output_dim == 3 * genes_dim:
            # Get ridge_lambda from the network object
            ridge = getattr(network, "ridge_lambda_for_loss", 0.0)
            loss_fn = PackedZINBLoss(ridge_lambda=ridge)
        else:
            # Fallback for models I didn't modify or non-std packing
            print(f"Warning: Packed layer found but output dim {output_dim} doesn't match 2*G or 3*G (G={genes_dim}). Falling back to WrappedLoss.")
            loss_fn = WrappedLoss(network.loss)

    model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=False, jit_compile=False)

    # FIX Retracing: Compile auxiliary models (encoder and extra models) explicitly.
    # This prevents lazy tracing when predict() is called later.
    if network.encoder:
        network.encoder.compile(run_eagerly=False, jit_compile=False)

    for m in network.extra_models.values():
        # Some extra_models might be functions (e.g., _disp), not Keras models.
        if isinstance(m, keras.Model):
            m.compile(run_eagerly=False, jit_compile=False)
    
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

    # Prepare data
    # Prepare data using the centralized method in the network object
    # This ensures consistency between training and prediction inputs.
    X, sf = network._prepare_inputs(adata)

    # FIX Retracing: Use a list/tuple matching the Model definition order: [count, size_factors]
    inputs = [X, sf]   

    y = adata.raw.X if use_raw_as_output else adata.X
    if sp.issparse(y):
        y_dense = y.toarray()
    else:
        y_dense = np.asarray(y)

    output = y_dense
    
    if output_subset:
        gene_idx = [np.where(adata.raw.var_names == x)[0][0] for x in output_subset]
        output = output[:, gene_idx]
        
    # FIX Retracing: Ensure output dtype is float32
    output = np.ascontiguousarray(output, dtype=np.float32)

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
    keras.utils.set_random_seed(42)
    os.environ["PYTHONHASHSEED"] = "0"

    # Fix: Clear Keras session to release resources from previous models and reset
    # the tracing history. This mitigates retracing warnings when iterating
    # over datasets with different shapes.
    gc.collect()
    keras.backend.clear_session()

    # if args.threads:
    #     try:
    #         tf.config.threading.set_intra_op_parallelism_threads(int(args.threads))
    #         tf.config.threading.set_inter_op_parallelism_threads(int(args.threads))
    #     except Exception:
    #         pass

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

    # Ensure the training data is a concrete copy, not a view.
    # Views can sometimes lead to subtle issues with memory layout.
    train_adata = adata[adata.obs.dca_split == "train"]
    if hasattr(train_adata, 'is_view') and train_adata.is_view:
        train_adata = train_adata.copy()

    losses = train(
        train_adata,
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
