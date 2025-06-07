"""A simple outlier detection class.

"""
__author__ = 'Paul Landes'

from typing import List, Union, Iterable
from dataclasses import dataclass, field
import itertools as it
import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from scipy import stats
from zensols.persist import persisted
from . import DatasetError


@dataclass
class OutlierDetector(object):
    """Simple outlier detection utility that provides a few differnt methods of
    calculation.  These include :meth:`z-score`, :meth:`mahalanobis` and
    :meth:`robust_mahalanobis`.

    This class removes either using a method specific :obj:`threshold` or by a
    :obj:`proportion` of the data set.

    """
    DETECTION_METHODS = frozenset({
        'z_score', 'mahalanobis', 'robust_mahalanobis'})

    data: Union[np.ndarray, pd.DataFrame] = field()
    """The dataframe on which to find outliers given the data.  Data points are
    rows and the feature vectors are columns.

    """
    default_method: str = field(default='mahalanobis')
    """The method used when invoking as a :class:`.Callable` with the
    :meth:`__call__` method.  This must be one of :obj:`DETECTION_METHODS`.

    """
    threshold: float = field(default=None)
    """The outlier threshold, which is method dependent.  This is ignored if
    :obj:`proportion` is set.

    """
    proportion: float = field(default=None)
    """The proportion of the dataset to use for outliers.  The higher the number
    the more outliers.

    :see: :obj:`threshold`

    """
    return_indicators: bool = field(default=None)
    """Whether to return a list of ``False`` (not outlier) or ``True`` (outlier)
    instead of indexes in to the input matrix/dataframe (:obj:`data`).

    """
    def __post_init__(self):
        if self.default_method not in self.DETECTION_METHODS:
            raise DatasetError(
                f'No such detection method: {self.default_method}')

    @property
    @persisted('_numpy')
    def numpy(self) -> np.ndarray:
        """The numpy form of :obj:`data`.  If :obj:`data` is a dataframe, it is
        converted to a numpy array.

        """
        return self._get_arr()

    def _get_arr(self) -> np.ndarray:
        data = self.data
        if isinstance(data, pd.DataFrame):
            data = self.data.to_numpy()
        return data

    def _to_indicators(self, indicies: np.ndarray) -> np.ndarray:
        """Convert row indexes in to a mask usable in :meth:`numpy.where`.

        :param indicies: row indexes in to :obj:`numpy`

        """
        # shape: (R, C)
        arr: np.ndarray = self.numpy
        mask: np.ndarray = np.repeat(False, arr.shape[0])
        for oix in indicies:
            mask[oix] = True
        return mask

    def _select_indicies(self, dists: Iterable[Union[int, float]],
                         threshold: Union[int, float]) -> np.ndarray:
        """Find outliers."""
        if self.proportion is None:
            threshold = threshold if self.threshold is None else self.threshold
            outliers: List[int] = []
            for i, v in enumerate(dists):
                if v > threshold:
                    outliers.append(i)
        else:
            drs = sorted(zip(dists, it.count()), key=lambda x: x[0])
            take = 1 - int(self.proportion * len(drs))
            outliers = sorted(map(lambda x: x[1], drs[take:]))
        if self.return_indicators:
            outliers = self._to_indicators(outliers)
        return outliers

    def z_score(self, column: Union[int, str]) -> np.ndarray:
        """Use a Z-score to detect anomolies.

        :param column: the column to use for the z-score analysis.

        :param threshold: the threshold above which a data point is considered
                          an outlier

        :return: indexes in to :obj:`data` rows (indexes of a dataframe) of the
                 outliers

        """
        if isinstance(column, str):
            if not isinstance(self.data, pd.DataFrame):
                raise DatasetError(
                    'Can not index numpy arrays as string column: {column}')
            column = self.data.columns.get_loc(column)
        # shape: (R, C)
        arr: np.ndarray = self.numpy
        z = np.abs(stats.zscore(arr[:, column]))
        return self._select_indicies(z, 3.)

    def _set_chi_threshold(self, sig: float) -> float:
        # shape: (R, C)
        arr: np.ndarray = self.numpy
        # degrees of freedom (df parameter) are number of variables
        C = np.sqrt(stats.chi2.ppf((1. - sig), df=arr.shape[1]))
        return C

    def mahalanobis(self, significance: float = 0.001) -> np.ndarray:
        """Detect outliers using the Mahalanbis distince in high dimension.

        Assuming a multivariate normal distribution of the data with K
        variables, the Mahalanobis distance follows a chi-squared distribution
        with K degrees of freedom.  For this reason, the cut-off is defined by
        the square root of the Chi^2 percent pointwise function.

        :param significance: 1 - the Chi^2 percent point function (inverse of
                             cdf / percentiles) outlier threshold; reasonable
                             values include 2.5%, 1%, 0.01%); if `None` use
                             :obj:`threshold` or :obj:`proportion`

        :return: indexes in to :obj:`data` rows (indexes of a dataframe) of the
                 outliers

        """
        # shape: (R, C)
        arr: np.ndarray = self.numpy
        # M-Distance, shape: (R,)
        x_minus_mu: pd.DataFrame = arr - np.mean(arr, axis=0)
        # covariance, shape: (C, C)
        cov: np.ndarray = np.cov(arr.T)
        # inverse covariance, shape: (C, C)
        inv_cov: np.ndarray = np.linalg.inv(cov)
        # shape: (R, C)
        left_term: np.ndarray = np.dot(x_minus_mu, inv_cov)
        # shape: (R, R)
        dist: np.ndarray = np.dot(left_term, x_minus_mu.T)
        # shape (R,)
        md: np.ndarray = np.sqrt(dist.diagonal())

        C = self._set_chi_threshold(significance)
        return self._select_indicies(md, C)

    def robust_mahalanobis(self, significance: float = 0.001,
                           random_state: int = 0) -> np.ndarray:
        """Like :meth:`mahalanobis` but use a robust mean and covarance matrix
        by sampling the dataset.

        :param significance: 1 - the Chi^2 percent point function (inverse of
                             cdf / percentiles) outlier threshold; reasonable
                             values include 2.5%, 1%, 0.01%); if `None` use
                             :obj:`threshold` or :obj:`proportion`

        :return: indexes in to :obj:`data` rows (indexes of a dataframe) of the
                 outliers

        """
        arr: np.ndarray = self.numpy
        # minimum covariance determinant
        rng = np.random.RandomState(random_state)
        # random sample of data, shape: (R, C)
        X: np.ndarray = rng.multivariate_normal(
            mean=np.mean(arr, axis=0),
            cov=np.cov(arr.T),
            size=arr.shape[0])
        # get robust estimates for the mean and covariance
        cov = MinCovDet(random_state=random_state).fit(X)
        # robust covariance metric; shape: (C, C)
        mcd: np.ndarray = cov.covariance_
        # robust mean, shape: (C,)
        rmean: np.ndarray = cov.location_
        # inverse covariance metric, shape: (C, C)
        inv_cov: np.ndarray = np.linalg.inv(mcd)
        # robust M-Distance, shape: (R, C)
        x_minus_mu: np.ndarray = arr - rmean
        # shape: (R, C)
        left_term: np.ndarray = np.dot(x_minus_mu, inv_cov)
        # m distance: shape: (R, R)
        dist: np.ndarray = np.dot(left_term, x_minus_mu.T)
        # distances: shape: (R,)
        md: np.ndarray = np.sqrt(dist.diagonal())

        C = self._set_chi_threshold(significance)
        return self._select_indicies(md, C)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Return the output of the method provided by :obj:`default_method`.
        All (keyword) arguments are passed on to the respective method.

        :return: indexes in to :obj:`data` rows (indexes of a dataframe) of the
                 outliers

        """
        meth = getattr(self, self.default_method)
        return meth(*args, **kwargs)
