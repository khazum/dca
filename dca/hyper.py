
import os
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from keras import optimizers as opt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import set_random_seed
try:
    import keras_tuner as kt  # package name: keras-tuner
except Exception as e:
    raise ImportError(
        "This feature requires the 'keras-tuner' package. "
        "Install it with: pip install keras-tuner"
    ) from e

from . import io
from .network import AE_types
from .train import WrappedLoss, PackedNBLoss  # use the same loss logic as training


@dataclass
class SearchSpace:
    # model space
    lr: Tuple[float, float] = (1e-4, 5e-3)  # log-uniform
    ridge: Tuple[float, float] = (1e-7, 1e-1)  # log-uniform
    l1_enc_coef: Tuple[float, float] = (1e-7, 1e-1)  # log-uniform
    hidden_size_choices: Tuple[Tuple[int, ...], ...] = (
        (64, 32, 64), (32, 16, 32), (64, 64), (32, 32), (16, 16),
        (16,), (32,), (64,), (128,)
    )
    activation_choices: Tuple[str, ...] = ('relu', 'selu', 'elu', 'PReLU', 'linear', 'LeakyReLU')
    aetype_choices: Tuple[str, ...] = ('zinb', 'zinb-conddisp')
    batchnorm_choices: Tuple[bool, ...] = (True, False)
    dropout_range: Tuple[float, float] = (0.0, 0.7)
    input_dropout_range: Tuple[float, float] = (0.0, 0.8)

    # data space
    norm_input_log_choices: Tuple[bool, ...] = (True, False)
    norm_input_zeromean_choices: Tuple[bool, ...] = (True, False)
    norm_input_sf_choices: Tuple[bool, ...] = (True, False)


class DCAHyperModel(kt.HyperModel):
    """Builds a Keras model given hp + dataset metadata."""
    def __init__(self, input_dim: int, debug: bool = False, optimizer_name: str = "RMSprop"):
        self.input_dim = input_dim
        self.debug = debug
        self.optimizer_name = optimizer_name

    def build(self, hp: kt.HyperParameters):
        # --- Model hyperparameters
        hidden_size = hp.Choice("hidden_size",
                                values=[",".join(map(str, hs)) for hs in SearchSpace().hidden_size_choices])
        hidden_size = tuple(int(x) for x in hidden_size.split(","))

        activation = hp.Choice("activation", SearchSpace().activation_choices)
        aetype = hp.Choice("aetype", SearchSpace().aetype_choices)
        batchnorm = hp.Choice("batchnorm", SearchSpace().batchnorm_choices)
        dropout = hp.Float("dropout", min_value=SearchSpace().dropout_range[0],
                           max_value=SearchSpace().dropout_range[1], step=0.05)
        input_dropout = hp.Float("input_dropout", min_value=SearchSpace().input_dropout_range[0],
                                 max_value=SearchSpace().input_dropout_range[1], step=0.05)
        ridge = hp.Float("ridge", min_value=np.log10(SearchSpace().ridge[0]),
                         max_value=np.log10(SearchSpace().ridge[1]), sampling="linear")
        l1_enc_coef = hp.Float("l1_enc_coef", min_value=np.log10(SearchSpace().l1_enc_coef[0]),
                               max_value=np.log10(SearchSpace().l1_enc_coef[1]), sampling="linear")
        # exponentiate logged params
        ridge = float(10.0 ** ridge)
        l1_enc_coef = float(10.0 ** l1_enc_coef)

        # --- Build network with the sampled hyperparameters
        net = AE_types[aetype](
            input_size=self.input_dim,
            hidden_size=hidden_size,
            l2_coef=0.0,
            l1_coef=0.0,
            l2_enc_coef=0.0,
            l1_enc_coef=l1_enc_coef,
            ridge=ridge,
            hidden_dropout=dropout,
            input_dropout=input_dropout,
            batchnorm=batchnorm,
            activation=activation,
            init="glorot_uniform",
            debug=self.debug,
        )
        net.build()

        # optimizer
        lr_log10 = hp.Float("lr_log10", min_value=np.log10(SearchSpace().lr[0]),
                            max_value=np.log10(SearchSpace().lr[1]), sampling="linear")
        lr = float(10.0 ** lr_log10)

        # Build optimizer from its name (e.g. "RMSprop")
        opt_cls = getattr(opt, self.optimizer_name)
        optimizer = opt_cls(learning_rate=lr, clipvalue=5.0)

        # Use same loss logic as training so we stay consistent across model types
        if any(l.name == "pack" for l in net.model.layers):
            loss_fn = PackedNBLoss()
        else:
            loss_fn = WrappedLoss(net.loss)

        net.model.compile(optimizer=optimizer, loss=loss_fn, run_eagerly=False, jit_compile=False)
        return net.model


class DCATuner(kt.BayesianOptimization):
    """Custom tuner that also tunes data preprocessing flags per trial."""
    def __init__(self, adata, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adata = adata

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters

        # --- Data hyperparameters
        norm_input_log = hp.Choice("norm_input_log", SearchSpace().norm_input_log_choices)
        norm_input_zeromean = hp.Choice("norm_input_zeromean", SearchSpace().norm_input_zeromean_choices)
        norm_input_sf = hp.Choice("norm_input_sf", SearchSpace().norm_input_sf_choices)

        # Prepare data according to trial's choices
        ad = self.adata.copy()
        ad = io.normalize(
            ad,
            size_factors=norm_input_sf,
            logtrans_input=norm_input_log,
            normalize_input=norm_input_zeromean,
        )

        # Keras expects dense arrays
        # Standardize X: Use toarray() instead of .A, enforce float32 and C-contiguous
        X_dense = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
        X = np.ascontiguousarray(X_dense, dtype=np.float32)

        # Standardize sf
        sf = np.asarray(ad.obs.size_factors).reshape(-1, 1)
        sf = np.ascontiguousarray(sf, dtype=np.float32)

        # Standardize y (output)
        y = ad.raw.X
        y_dense = y.toarray() if sp.issparse(y) else np.asarray(y)
        y = np.ascontiguousarray(y_dense, dtype=np.float32)

        # Feed tensors
        x_train = {"count": X, "size_factors": sf}
        y_train = y

        # Pass prepared data to fit()
        callbacks = fit_kwargs.pop("callbacks", [])
        # keep sensible defaults
        callbacks = list(callbacks) + [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=0),
        ]

        super().run_trial(
            trial,
            x=x_train,
            y=y_train,
            callbacks=callbacks,
            **fit_kwargs,
        )


def hyper(args):
    """Hyperparameter search using KerasTuner (Bayesian optimization).

    Results are saved under <outputdir>/ktune_results/:
      - best_hparams.json : best hyperparameters (including preprocessing flags)
      - best_model.keras  : best model weights in the Keras v3 format
      - trial_summary.txt : textual summary of trials
    """
    # Reproducibility
    np.random.seed(42)
    set_random_seed(42)
    os.environ["PYTHONHASHSEED"] = "0"

    # if args.threads:
    #     try:
    #         tf.config.threading.set_intra_op_parallelism_threads(int(args.threads))
    #         tf.config.threading.set_inter_op_parallelism_threads(int(args.threads))
    #     except Exception:
    #         pass

    # Load dataset once; tuning will handle per-trial normalization
    adata = io.read_dataset(
        args.input,
        transpose=(not args.transpose),  # assume gene x cell by default
        test_split=False,
        check_counts=args.checkcounts,
    )

    output_dir = os.path.join(args.outputdir, "ktune_results")
    os.makedirs(output_dir, exist_ok=True)

    # Build HyperModel (input_dim is number of genes)
    input_dim = adata.n_vars
    hypermodel = DCAHyperModel(input_dim=input_dim, debug=args.debug, optimizer_name=args.optimizer)

    tuner = DCATuner(
        adata=adata,
        hypermodel=hypermodel,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=int(args.hypern),
        directory=output_dir,
        project_name="dca_hp",
        overwrite=True,
        seed=42,
    )

    # Perform search
    tuner.search(
        epochs=int(args.hyperepoch),
        validation_split=0.2,
        batch_size=int(args.batchsize),
        verbose=1,
    )

    # Save best hyperparameters and model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    with open(os.path.join(output_dir, "best_hparams.json"), "w") as f:
        json.dump(best_hp.values, f, indent=2, sort_keys=True)

    best_model = tuner.get_best_models(num_models=1)[0]
    # Save in Keras .keras format
    best_model.save(os.path.join(output_dir, "best_model.keras"), include_optimizer=False)

    # Write textual summary
    with open(os.path.join(output_dir, "trial_summary.txt"), "w") as f:
        tuner.results_summary(num_trials=10, print_fn=lambda s: f.write(s + "\n"))

    print("KerasTuner search finished. Results written to:", output_dir)
