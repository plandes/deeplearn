"""Dimension reduction wrapper and utility classes.

"""
__author__ = 'Paul Landes'

from typing import Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from zensols.util import APIError
from zensols.config import Dictable
from zensols.persist import persisted

logger = logging.getLogger(__name__)


@dataclass
class DimensionReducer(Dictable):
    """Reduce the dimensionality of a dataset.

    """
    _DICTABLE_ATTRIBUTES = {'n_points'}

    data: np.ndarray = field(repr=False)
    """The data that will be dimensionally reduced."""

    dim: int = field()
    """The lowered dimension spaace."""

    reduction_meth: str = field(default='pca')
    """One of ``pca``, ``svd``, or ``tsne``."""

    normalize: str = field(default='unit')
    """One of:

      * ``unit``: normalize to unit vectors

      * ``standardize``: standardize by removing the mean and scaling to unit
                         variance

      * ``None``: make no modifications to the data

    """
    model_args: Dict[str, Any] = field(default_factory=dict)
    """Additional kwargs to pass to the model initializer."""

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.normalize == 'standarize':
            x = StandardScaler().fit_transform(data)
        elif self.normalize == 'unit':
            x = normalize(data)
        return x

    @persisted('_dim_reduced')
    def _dim_reduce(self) -> np.ndarray:
        model = None
        data = self.data
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'using {self.reduction_meth} ({self.dim}) ' +
                         f'on {data.shape}')
        if self.normalize:
            if self.normalize == 'standardize':
                data = StandardScaler().fit_transform(data)
            elif self.normalize == 'unit':
                data = normalize(data)
            else:
                raise APIError(
                    f'Unknown normalization method: {self.normalize}')
        if self.reduction_meth == 'pca':
            model = PCA(self.dim, **self.model_args)
            data = model.fit_transform(data)
        elif self.reduction_meth == 'svd':
            model = TruncatedSVD(self.dim, **self.model_args)
            data = model.fit_transform(data)
        elif self.reduction_meth == 'tsne':
            if data.shape[-1] > 50:
                data = PCA(50).fit_transform(data)
            params = dict(init='pca', learning_rate='auto')
            params.update(self.model_args)
            model = TSNE(self.dim, **params)
            data = model.fit_transform(data)
        else:
            raise APIError('Unknown dimension reduction method: ' +
                           self.reduction_meth)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'reduced shape: {data.shape}')
        return data, model

    @property
    def n_points(self) -> Tuple[int]:
        return self.data.shape[0]

    @property
    @persisted('_reduced')
    def reduced(self) -> np.ndarray:
        return self._dim_reduce()[0]

    @property
    def model(self) -> Union[PCA, TruncatedSVD, TSNE]:
        return self._dim_reduce()[1]

    def _get_reduced_data(self, data: np.ndarray) -> np.ndarray:
        data: np.ndarray = self.reduced if data is None else data
        if data.shape[-1] != self.data.shape[-1]:
            X = self.model.inverse_transform(data)
        else:
            X: np.ndarray = data
        return X


@dataclass
class DecomposeDimensionReducer(DimensionReducer):
    """A dimensionality reducer that uses eigenvector decomposition such as PCA
    or SVD.

    """
    _DICTABLE_ATTRIBUTES = DimensionReducer._DICTABLE_ATTRIBUTES | \
        {'description'}

    def __post_init__(self):
        assert self.is_decompose_method(self.reduction_meth)

    @staticmethod
    def is_decompose_method(reduction_meth: str) -> bool:
        """Return whether the reduction is a decomposition method.

        :see: :obj:`reduction_meth`

        """
        return reduction_meth == 'pca' or reduction_meth == 'svd'

    def get_components(self, data: np.ndarray = None,
                       one_dir: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Create a start and end points that make the PCA component, which is
        useful for rendering lines for visualization.

        :param: use in place of the :obj:`data` for component calculation using
                the (already) trained model

        :param one_dir: whether or not to create components one way from the
                        mean, or two way (forward and backward) from the mean


        :return: a tuple of numpy arrays, each as a start and end stacked for
                 each component

        """
        comps: List[np.ndarray] = []
        X = self._get_reduced_data(data)
        # fit a covariance matrix on the data
        cov_matrix: np.ndarray = np.cov(X.T)
        # find the center from where the PCA component starts
        trans_mean: np.ndarray = data.mean(axis=0)
        # the components of the model are the eigenvectors of the covarience
        # matrix
        evecs: np.ndarray = self.model.components_
        # the eigenvalues of the covariance matrix
        evs: np.ndarray = self.model.explained_variance_
        for n_comp, (eigenvector, eigenvalue) in enumerate(zip(evecs, evs)):
            # map a data point as a component back to the original data space
            end: np.ndarray = np.dot(cov_matrix, eigenvector) / eigenvalue
            # map to the reduced dimensional space
            end = self.model.transform([end])[0]
            start = trans_mean
            if not one_dir:
                # make the component "double sided"
                start = start - end
            comps.append(np.stack((start, end)))
        return comps

    @property
    def description(self) -> Dict[str, Any]:
        """A object graph of data that describes the results of the model."""
        tot_ev = 0
        model = self.model
        evs = []
        for i, ev in enumerate(model.explained_variance_ratio_):
            evs.append(ev)
            tot_ev += ev
        noise: float = None
        if hasattr(model, 'noise_variance_'):
            noise = model.noise_variance_
        return {'components': len(model.components_),
                'noise': noise,
                'total_variance': tot_ev,
                'explained_varainces': evs}
