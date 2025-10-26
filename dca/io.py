# Copyright 2017 Goekcen Eraslan
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

import pickle

import numpy as np
import scipy.sparse as sp_sparse
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

def read_dataset(adata, transpose=False, test_split=False, copy=False, check_counts=True):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata, first_column_names=True)
    else:
        raise NotImplementedError

    if check_counts:
        # check if observations are unnormalized using first 10
        X_subset = adata.X[:10]
        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        if sp_sparse.issparse(X_subset):
            assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
        else:
            assert np.all(X_subset.astype(int) == X_subset), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')
    print('dca: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    # Keep a raw copy of counts prior to any normalization
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        # Compute per-cell library sizes (float) before normalization
        if sp_sparse.issparse(adata.X):
            n_counts = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            n_counts = np.asarray(adata.X.sum(axis=1)).ravel()
        n_counts = n_counts.astype(np.float64, copy=False)
        adata.obs['n_counts'] = n_counts

        # Size factors: library size / median library size
        adata.obs['size_factors'] = n_counts / np.median(n_counts)

        # Ensure float dtype prior to per-cell scaling to avoid casting issues
        adata.X = adata.X.astype(np.float32)

        # Modern Scanpy normalization (replaces deprecated normalize_per_cell)
        try:
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True, key_added=None)
        except TypeError:
            sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    else:
        adata.obs['size_factors'] = np.ones(adata.n_obs, dtype=np.float32)

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('dca: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))
