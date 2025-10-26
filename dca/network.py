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

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda, Multiply, Concatenate
from keras.models import Model
from keras.regularizers import l1_l2
from keras.losses import MeanSquaredError
from keras import ops
import scipy.sparse as sp

from .loss import poisson_loss, NB, ZINB
from .layers import (
    ConstantDispersionLayer,
    SliceLayer,
    ElementwiseDense,
)
from .io import write_text_matrix

def maybe_l1_l2(l1, l2):
    return l1_l2(l1, l2) if (l1 and l1 > 0) or (l2 and l2 > 0) else None

MeanAct = lambda x: ops.clip(ops.exp(x), 1e-5, 1e6)
DispAct = lambda x: ops.clip(ops.softplus(x), 1e-4, 1e4)

advanced_activations = ("PReLU", "LeakyReLU")


class Autoencoder:
    def __init__(
        self,
        input_size,
        output_size=None,
        hidden_size=(64, 32, 64),
        l2_coef=0.0,
        l1_coef=0.0,
        l2_enc_coef=0.0,
        l1_enc_coef=0.0,
        ridge=0.0,
        hidden_dropout=0.0,
        input_dropout=0.0,
        batchnorm=True,
        activation="relu",
        init="glorot_uniform",
        file_path=None,
        debug=False,
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.loss = None
        self.file_path = file_path
        self.extra_models = {}
        self.model = None
        self.encoder = None
        self.decoder = None
        self.input_layer = None
        self.sf_layer = None
        self.debug = debug

        if self.output_size is None:
            self.output_size = input_size

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout] * len(self.hidden_size)

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name="count")
        self.sf_layer = Input(shape=(1,), name="size_factors")
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name="input_dropout")(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(
            zip(self.hidden_size, self.hidden_dropout)
        ):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = "center"
                stage = "center"  # let downstream know where we are
            elif i < center_idx:
                layer_name = "enc%s" % i
                stage = "encoder"
            else:
                layer_name = "dec%s" % (i - center_idx)
                stage = "decoder"

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0.0 and stage in ("center", "encoder"):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0.0 and stage in ("center", "encoder"):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            last_hidden = Dense(
                hid_size,
                activation=None,
                kernel_initializer=self.init,
                kernel_regularizer=maybe_l1_l2(l1, l2),
                name=layer_name,
            )(last_hidden)
            if self.batchnorm:
                last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)

            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            if self.activation in advanced_activations:
                last_hidden = keras.layers.__dict__[self.activation](
                    name="%s_act" % layer_name
                )(last_hidden)
            else:
                last_hidden = Activation(self.activation, name="%s_act" % layer_name)(
                    last_hidden
                )

            if hid_drop > 0.0:
                last_hidden = Dropout(hid_drop, name="%s_drop" % layer_name)(
                    last_hidden
                )

        self.decoder_output = last_hidden
        self.build_output()

    def build_output(self):

        self.loss = MeanSquaredError()
        mean = Dense(
            self.output_size,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        output = Multiply(name="output_scaled")([mean, self.sf_layer])

        # keep unscaled output as an extra model
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            with open(os.path.join(self.file_path, "model.pickle"), "wb") as f:
                pickle.dump(self, f)

    def load_weights(self, filename):
        self.model.load_weights(filename)
        self.encoder = self.get_encoder()
        self.decoder = None  # get_decoder()

    def get_decoder(self):
        i = 0
        for l in self.model.layers:
            if l.name == "center_drop":
                break
            i += 1

        return Model(
            inputs=self.model.get_layer(index=i + 1).input, outputs=self.model.output
        )

    def get_encoder(self, activation=False):
        if activation:
            output_tensor = self.model.get_layer("center_act").output
        else:
            output_tensor = self.model.get_layer("center").output

        # Optimization: Define the encoder model using only the count input layer (self.input_layer).
        # The encoding path does not depend on size factors.
        ret = Model(
            inputs=self.input_layer,
            outputs=output_tensor,
            name="encoder_model"
        )
        return ret

    def _prepare_inputs(self, adata):
        # Centralized input preparation (Handles FIX Retracing logic)
        # Ensure inputs are consistently dense, float32, and C-contiguous.
        
        # Prepare size factors (sf)
        # Use np.asarray for robust Pandas conversion, then ascontiguousarray.
        sf = np.asarray(adata.obs.size_factors).reshape(-1, 1)
        sf = np.ascontiguousarray(sf, dtype=np.float32)
        
        # Prepare input features (X)
        # Standardize X: Use sp.issparse() and .toarray()
        if sp.issparse(adata.X):
             X_dense = adata.X.toarray()
        else:
             X_dense = np.asarray(adata.X)
        X = np.ascontiguousarray(X_dense, dtype=np.float32)
        
        return X, sf

    def predict(self, adata, mode="denoise", return_info=False, copy=False, batch_size=256):
        assert mode in ("denoise", "latent", "full"), "Unknown mode"

        adata = adata.copy() if copy else adata

        # Optimization: Prepare inputs once
        X, sf = self._prepare_inputs(adata)
        inputs = [X, sf]

        # Optimization: Allow subclasses to perform predictions using extra models if return_info=True
        if return_info:
            self._predict_info(adata, X, sf, batch_size)

        if mode in ("latent", "full"):
            print("dca: Calculating low dimensional representations...")
            # Optimization: The encoder only requires the count input (X)
            adata.obsm["X_dca"] = self.encoder.predict(X, batch_size=batch_size, verbose=0)                
        if mode in ("denoise", "full"):
            print("dca: Calculating reconstructions...")
            pred = self.model.predict(inputs, batch_size=batch_size, verbose=0)
            
            # Check if a layer named "pack" exists (the Concatenate layer now has this name)
            if any(l.name == "pack" for l in self.model.layers):
                # ... (rest of the predict method handling packed output remains the same)
                g = self.output_size
                n_features = pred.shape[-1]
                
                if n_features == g:
                    adata.X = pred
                elif n_features == 2 * g:
                    adata.X = pred[:, :g]  # Keep mu
                elif n_features == 3 * g:
                    adata.X = pred[:, :g]  # Keep mu
                else:
                    raise ValueError(
                        f"Packed output width {n_features} is not 1, 2, or 3 times output_size ({g})."
                    )
            else:
                # Non-packed models
                adata.X = pred
        if mode == "latent":
            adata.X = adata.raw.X.copy()  # recover normalized expression values

        return adata if copy else None

    def _predict_info(self, adata, X, sf, batch_size):
        # Base implementation does nothing. Subclasses override this.
        pass

    def write(self, adata, file_path, mode="denoise", colnames=None):

        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        print("dca: Saving output(s)...")
        os.makedirs(file_path, exist_ok=True)

        if mode in ("denoise", "full"):
            print("dca: Saving denoised expression...")
            write_text_matrix(
                adata.X,
                os.path.join(file_path, "mean.tsv"),
                rownames=rownames,
                colnames=colnames,
                transpose=True,
            )

        if mode in ("latent", "full"):
            print("dca: Saving latent representations...")
            write_text_matrix(
                adata.obsm["X_dca"],
                os.path.join(file_path, "latent.tsv"),
                rownames=rownames,
                transpose=False,
            )


class PoissonAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        self.loss = poisson_loss

        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()


class NBConstantDispAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        # 2. Scaled mean (mu)
        # OPTIMIZATION: Replace ColwiseMultLayer
        output = Multiply(name="output_scaled")([mean, self.sf_layer])

        # FIX Gradient Flow: Use the redesigned ConstantDispersionLayer
        # 3. Dispersion (theta)
        disp_layer = ConstantDispersionLayer(name="dispersion")
        # Pass 'output' (B, G) for batch size inference. This ensures gradient flow.
        theta_bcast = disp_layer(output)

        # 4. Pack [μ | θ]
        # OPTIMIZATION: Replace Packing Lambda with native Concatenate
        packed = Concatenate(axis=-1, name="pack")([output, theta_bcast])

        # Let train.py choose PackedNBLoss
        self.loss = None

        # Introspection helpers (updated to access the raw weight)
        def _disp():
            import numpy as np
            # Access the raw weight (theta_raw) which is the first weight
            try:
                # We access the weight by index 0 (theta_raw)
                theta_w_raw = self.model.get_layer("dispersion").get_weights()[0]
                return np.squeeze(np.clip(np.exp(theta_w_raw), 1e-3, 1e4))
            except Exception as e:
                print(f"dca: Warning: Could not retrieve dispersion weights: {e}")
                return None
                
        self.extra_models["dispersion"] = _disp
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        # Model now outputs packed [μ | θ]
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)
        self.encoder = self.get_encoder()

    # Optimization: Override _predict_info instead of predict
    def _predict_info(self, adata, X, sf, batch_size):
        # Constant dispersion doesn't depend on input X, retrieve from layer weights
        # We assign the dispersion values (gene-wise constants) to adata.var
        disp_values = self.extra_models["dispersion"]()
        if disp_values is not None:
            # Robustness: Ensure the shape matches the number of genes in the current adata view/copy
            if len(disp_values) == adata.n_vars:
                adata.var["X_dca_dispersion"] = disp_values
            else:
                print(f"dca: Warning: Mismatch between model dispersion parameters ({len(disp_values)}) and adata genes ({adata.n_vars}). Skipping dispersion output.")

    def write(self, adata, file_path, mode="denoise", colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode, colnames=colnames)
        if "X_dca_dispersion" in adata.var_keys():
            write_text_matrix(
                adata.var["X_dca_dispersion"].reshape(1, -1),
                os.path.join(file_path, "dispersion.tsv"),
                colnames=colnames,
                transpose=True,
            )


class NBAutoencoder(Autoencoder):

    def build_output(self):
        disp = Dense(
            self.output_size,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.decoder_output)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        packed = Concatenate(axis=-1, name="pack")([output, disp])


        # keep for inspection; training will override self.loss
        # Let train.py choose PackedNBLoss
        self.loss = None
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)

    # Optimization: Override _predict_info instead of predict
    def _predict_info(self, adata, X, sf, batch_size):
        # Conditional dispersion depends on input X
        # Predict dispersion using the standardized input features (X) prepared by Autoencoder.predict
        adata.obsm["X_dca_dispersion"] = self.extra_models["dispersion"].predict(X, batch_size=batch_size, verbose=0)

    def write(self, adata, file_path, mode="denoise", colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        super().write(adata, file_path, mode, colnames=colnames)

        if "X_dca_dispersion" in adata.obsm_keys():
            write_text_matrix(
                adata.obsm["X_dca_dispersion"],
                os.path.join(file_path, "dispersion.tsv"),
                colnames=colnames,
                transpose=True,
            )


class NBSharedAutoencoder(NBAutoencoder):

    def build_output(self):
        disp = Dense(
            1,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.decoder_output)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        # Broadcast disp from (B, 1) to (B, G)
        disp_bcast = keras.layers.Lambda(
             lambda t: t[0] * ops.ones_like(t[1]),
             name="broadcast_disp"
        )([disp, mean])

        # Pack [mu | theta]
        packed = Concatenate(axis=-1, name="pack")([output, disp_bcast])
        self.loss = None # Let train.py choose PackedNBLoss
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)
        self.encoder = self.get_encoder()


class ZINBAutoencoder(Autoencoder):

    def build_output(self):
        pi = Dense(
            self.output_size,
            activation="sigmoid",
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="pi",
        )(self.decoder_output)

        disp = Dense(
            self.output_size,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.decoder_output)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        
        packed = Concatenate(axis=-1, name="pack")([output, disp, pi])
        
        # Store ridge_lambda for the loss constructor in train.py
        self.ridge_lambda_for_loss = self.ridge 
        self.loss = None # This will be replaced in dca/train.py

        self.extra_models["pi"] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)

        self.encoder = self.get_encoder()

    def predict(
        self, adata, mode="denoise", return_info=False, copy=False, batch_size=256, colnames=None
    ):
        # The signature must match the base class or be compatible if called directly.
        # We adjust the call to super() to include the batch_size parameter.
        return super().predict(adata, mode, return_info, copy, batch_size=batch_size)

    # Optimization: Override _predict_info instead
    def _predict_info(self, adata, X, sf, batch_size):
        # Use the standardized input features (X) prepared by Autoencoder.predict
        adata.obsm["X_dca_dispersion"] = self.extra_models["dispersion"].predict(X, batch_size=batch_size, verbose=0)
        adata.obsm["X_dca_dropout"] = self.extra_models["pi"].predict(X, batch_size=batch_size, verbose=0)

    def write(self, adata, file_path, mode="denoise", colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode, colnames=colnames)

        if "X_dca_dispersion" in adata.obsm_keys():
            write_text_matrix(
                adata.obsm["X_dca_dispersion"],
                os.path.join(file_path, "dispersion.tsv"),
                colnames=colnames,
                transpose=True,
            )

        if "X_dca_dropout" in adata.obsm_keys():
            write_text_matrix(
                adata.obsm["X_dca_dropout"],
                os.path.join(file_path, "dropout.tsv"),
                colnames=colnames,
                transpose=True,
            )


class ZINBAutoencoderElemPi(ZINBAutoencoder):
    def __init__(self, sharedpi=False, **kwds):
        super().__init__(**kwds)
        self.sharedpi = sharedpi

    def build_output(self):
        disp = Dense(
            self.output_size,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.decoder_output)

        mean_no_act = Dense(
            self.output_size,
            activation=None,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean_no_act",
        )(self.decoder_output)

        minus = Lambda(lambda x: -x)
        mean_no_act_neg = minus(mean_no_act)
        pidim = self.output_size if not self.sharedpi else 1

        pi = ElementwiseDense(
            pidim,
            activation="sigmoid",
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="pi",
        )(mean_no_act_neg)

        mean = Activation(MeanAct, name="mean")(mean_no_act)

        # Handle sharedpi broadcast
        if self.sharedpi:
            pi_bcast = keras.layers.Lambda(
                lambda t: t[0] * ops.ones_like(t[1]),  # (B,1) -> (B,G)
                name="broadcast_pi"
            )([pi, mean])
        else:
            pi_bcast = pi

        output = Multiply(name="output_scaled")([mean, self.sf_layer])

        packed = Concatenate(axis=-1, name="pack")([output, disp, pi_bcast]) # pi_bcast is (B, G)

        self.ridge_lambda_for_loss = self.ridge 
        self.loss = None

        self.extra_models["pi"] = Model(inputs=self.input_layer, outputs=pi) # Original pi
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)

        self.encoder = self.get_encoder()


class ZINBSharedAutoencoder(ZINBAutoencoder):

    def build_output(self):
        pi = Dense(
            1,
            activation="sigmoid",
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="pi",
        )(self.decoder_output)

        disp = Dense(
            1,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.decoder_output)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)
        
        # Need to broadcast pi and disp from (B, 1) to (B, G)
        pi_bcast = keras.layers.Lambda(
            lambda t: t[0] * ops.ones_like(t[1]),
            name="broadcast_pi"
        )([pi, mean])
        disp_bcast = keras.layers.Lambda(
            lambda t: t[0] * ops.ones_like(t[1]),
            name="broadcast_disp"
        )([disp, mean])

        output = Multiply(name="output_scaled")([mean, self.sf_layer])

        packed = Concatenate(axis=-1, name="pack")([output, disp_bcast, pi_bcast])

        self.ridge_lambda_for_loss = self.ridge 
        self.loss = None
        
        self.extra_models["pi"] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)
        self.encoder = self.get_encoder()


class ZINBConstantDispAutoencoder(Autoencoder):

    def build_output(self):
        pi = Dense(
            self.output_size,
            activation="sigmoid",
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="pi",
        )(self.decoder_output)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.decoder_output)

        # 3. Scaled Mean (mu)
        # OPTIMIZATION: Replace ColwiseMultLayer
        output = Multiply(name="output_scaled")([mean, self.sf_layer])

        # FIX Gradient Flow: Use the redesigned ConstantDispersionLayer
        # 4. Dispersion (theta)
        disp_layer = ConstantDispersionLayer(name="dispersion")
        # Use 'output' (B, G) for batch size inference and ensure gradient flow.
        theta_bcast = disp_layer(output)

        # 5. Pack [mu, theta, pi]
        # OPTIMIZATION: Replace Packing Lambda
        packed = Concatenate(axis=-1, name="pack")([output, theta_bcast, pi])
        
        self.ridge_lambda_for_loss = self.ridge 
        self.loss = None

        # Introspection helpers (updated)
        self.extra_models["pi"] = Model(inputs=self.input_layer, outputs=pi)
        def _disp():
            import numpy as np
            try:
                # Access the weight by index 0 (theta_raw)
                theta_w_raw = self.model.get_layer("dispersion").get_weights()[0]
                return np.squeeze(np.clip(np.exp(theta_w_raw), 1e-3, 1e4))
            except Exception as e:
                print(f"dca: Warning: Could not retrieve dispersion weights: {e}")
                return None
        self.extra_models["dispersion"] = _disp
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models["decoded"] = Model(
            inputs=self.input_layer, outputs=self.decoder_output
        )

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)
        self.encoder = self.get_encoder()

    # Optimization: Override _predict_info instead of predict
    def _predict_info(self, adata, X, sf, batch_size):
        # 1. Constant dispersion retrieval (with robustness checks)
        disp_values = self.extra_models["dispersion"]()
        if disp_values is not None:
            if len(disp_values) == adata.n_vars:
                adata.var["X_dca_dispersion"] = disp_values
            else:
                print(f"dca: Warning: Mismatch between model dispersion parameters ({len(disp_values)}) and adata genes ({adata.n_vars}). Skipping dispersion output.")
        
        # 2. Conditional dropout (pi) prediction
        # Use the standardized input features (X)
        adata.obsm["X_dca_dropout"] = self.extra_models["pi"].predict(X, batch_size=batch_size, verbose=0)

    def write(self, adata, file_path, mode="denoise", colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode)

        if "X_dca_dispersion" in adata.var_keys():
            write_text_matrix(
                adata.var["X_dca_dispersion"].values.reshape(1, -1),
                os.path.join(file_path, "dispersion.tsv"),
                colnames=colnames,
                transpose=True,
            )

        if "X_dca_dropout" in adata.obsm_keys():
            write_text_matrix(
                adata.obsm["X_dca_dropout"],
                os.path.join(file_path, "dropout.tsv"),
                colnames=colnames,
                transpose=True,
            )


class ZINBForkAutoencoder(ZINBAutoencoder):

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name="count")
        self.sf_layer = Input(shape=(1,), name="size_factors")
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name="input_dropout")(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(
            zip(self.hidden_size, self.hidden_dropout)
        ):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = "center"
                stage = "center"  # let downstream know where we are
            elif i < center_idx:
                layer_name = "enc%s" % i
                stage = "encoder"
            else:
                layer_name = "dec%s" % (i - center_idx)
                stage = "decoder"

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0.0 and stage in ("center", "encoder"):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0.0 and stage in ("center", "encoder"):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            if i > center_idx:
                self.last_hidden_mean = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name="%s_last_mean" % layer_name,
                )(last_hidden)
                self.last_hidden_disp = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name="%s_last_disp" % layer_name,
                )(last_hidden)
                self.last_hidden_pi = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name="%s_last_pi" % layer_name,
                )(last_hidden)

                if self.batchnorm:
                    self.last_hidden_mean = BatchNormalization(
                        center=True, scale=False
                    )(self.last_hidden_mean)
                    self.last_hidden_disp = BatchNormalization(
                        center=True, scale=False
                    )(self.last_hidden_disp)
                    self.last_hidden_pi = BatchNormalization(center=True, scale=False)(
                        self.last_hidden_pi
                    )

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                self.last_hidden_mean = Activation(
                    self.activation, name="%s_mean_act" % layer_name
                )(self.last_hidden_mean)
                self.last_hidden_disp = Activation(
                    self.activation, name="%s_disp_act" % layer_name
                )(self.last_hidden_disp)
                self.last_hidden_pi = Activation(
                    self.activation, name="%s_pi_act" % layer_name
                )(self.last_hidden_pi)

                if hid_drop > 0.0:
                    self.last_hidden_mean = Dropout(
                        hid_drop, name="%s_mean_drop" % layer_name
                    )(self.last_hidden_mean)
                    self.last_hidden_disp = Dropout(
                        hid_drop, name="%s_disp_drop" % layer_name
                    )(self.last_hidden_disp)
                    self.last_hidden_pi = Dropout(
                        hid_drop, name="%s_pi_drop" % layer_name
                    )(self.last_hidden_pi)

            else:
                last_hidden = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name=layer_name,
                )(last_hidden)

                if self.batchnorm:
                    last_hidden = BatchNormalization(center=True, scale=False)(
                        last_hidden
                    )

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                if self.activation in advanced_activations:
                    last_hidden = keras.layers.__dict__[self.activation](
                        name="%s_act" % layer_name
                    )(last_hidden)
                else:
                    last_hidden = Activation(
                        self.activation, name="%s_act" % layer_name
                    )(last_hidden)

                if hid_drop > 0.0:
                    last_hidden = Dropout(hid_drop, name="%s_drop" % layer_name)(
                        last_hidden
                    )

        self.build_output()

    def build_output(self):
        pi = Dense(
            self.output_size,
            activation="sigmoid",
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="pi",
        )(self.last_hidden_pi)

        disp = Dense(
            self.output_size,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.last_hidden_disp)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.last_hidden_mean)

        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        
        packed = Concatenate(axis=-1, name="pack")([output, disp, pi])
        
        self.ridge_lambda_for_loss = self.ridge 
        self.loss = None

        self.extra_models["pi"] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)

        self.encoder = self.get_encoder()


class NBForkAutoencoder(NBAutoencoder):

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name="count")
        self.sf_layer = Input(shape=(1,), name="size_factors")
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name="input_dropout")(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(
            zip(self.hidden_size, self.hidden_dropout)
        ):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = "center"
                stage = "center"  # let downstream know where we are
            elif i < center_idx:
                layer_name = "enc%s" % i
                stage = "encoder"
            else:
                layer_name = "dec%s" % (i - center_idx)
                stage = "decoder"

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0.0 and stage in ("center", "encoder"):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0.0 and stage in ("center", "encoder"):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            if i > center_idx:
                self.last_hidden_mean = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name="%s_last_mean" % layer_name,
                )(last_hidden)
                self.last_hidden_disp = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name="%s_last_disp" % layer_name,
                )(last_hidden)

                if self.batchnorm:
                    self.last_hidden_mean = BatchNormalization(
                        center=True, scale=False
                    )(self.last_hidden_mean)
                    self.last_hidden_disp = BatchNormalization(
                        center=True, scale=False
                    )(self.last_hidden_disp)

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                self.last_hidden_mean = Activation(
                    self.activation, name="%s_mean_act" % layer_name
                )(self.last_hidden_mean)
                self.last_hidden_disp = Activation(
                    self.activation, name="%s_disp_act" % layer_name
                )(self.last_hidden_disp)

                if hid_drop > 0.0:
                    self.last_hidden_mean = Dropout(
                        hid_drop, name="%s_mean_drop" % layer_name
                    )(self.last_hidden_mean)
                    self.last_hidden_disp = Dropout(
                        hid_drop, name="%s_disp_drop" % layer_name
                    )(self.last_hidden_disp)

            else:
                last_hidden = Dense(
                    hid_size,
                    activation=None,
                    kernel_initializer=self.init,
                    kernel_regularizer=l1_l2(l1, l2),
                    name=layer_name,
                )(last_hidden)

                if self.batchnorm:
                    last_hidden = BatchNormalization(center=True, scale=False)(
                        last_hidden
                    )

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                if self.activation in advanced_activations:
                    last_hidden = keras.layers.__dict__[self.activation](
                        name="%s_act" % layer_name
                    )(last_hidden)
                else:
                    last_hidden = Activation(
                        self.activation, name="%s_act" % layer_name
                    )(last_hidden)

                if hid_drop > 0.0:
                    last_hidden = Dropout(hid_drop, name="%s_drop" % layer_name)(
                        last_hidden
                    )

        self.build_output()

    def build_output(self):

        disp = Dense(
            self.output_size,
            activation=DispAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="dispersion",
        )(self.last_hidden_disp)

        mean = Dense(
            self.output_size,
            activation=MeanAct,
            kernel_initializer=self.init,
            kernel_regularizer=maybe_l1_l2(self.l1_coef, self.l2_coef),
            name="mean",
        )(self.last_hidden_mean)

        output = Multiply(name="output_scaled")([mean, self.sf_layer])
        # Pack [mu | theta]
        packed = Concatenate(axis=-1, name="pack")([output, disp])
        self.loss = None # Let train.py choose PackedNBLoss
        self.extra_models["dispersion"] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models["mean_norm"] = Model(inputs=self.input_layer, outputs=mean)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=packed)
        self.encoder = self.get_encoder()


AE_types = {
    "normal": Autoencoder,
    "poisson": PoissonAutoencoder,
    "nb": NBConstantDispAutoencoder,
    "nb-conddisp": NBAutoencoder,
    "nb-shared": NBSharedAutoencoder,
    "nb-fork": NBForkAutoencoder,
    "zinb": ZINBConstantDispAutoencoder,
    "zinb-conddisp": ZINBAutoencoder,
    "zinb-shared": ZINBSharedAutoencoder,
    "zinb-fork": ZINBForkAutoencoder,
    "zinb-elempi": ZINBAutoencoderElemPi,
}
