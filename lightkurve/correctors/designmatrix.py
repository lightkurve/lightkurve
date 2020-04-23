"""Defines `DesignMatrix` and `DesignMatrixCollection`.

These classes are intended to make linear regression problems with a large
design matrix more easy.
"""
import warnings

from numba import jit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack, issparse, find, csc_matrix
from copy import deepcopy

from .. import MPLSTYLE
from ..utils import LightkurveWarning, plot_image


__all__ = ['DesignMatrix', 'DesignMatrixCollection']


class DesignMatrix():
    """A matrix of column vectors for use in linear regression.

    The purpose of this class is to provide a convenient method to interact
    with a set of one or more regressors which are known to correlate with
    trends or systematic noise signals which we want to remove from a light
    curve. Specifically, this class is designed to provide the design matrix
    for use by Lightkurve's `.RegressionCorrector` class.

    Parameters
    ----------
    X : dict, array, or `pandas.DataFrame` object
        Columns to include in the design matrix.  If this object is not a
        `~pandas.DataFrame` then it will be passed to the DataFrame constructor.
    columns : iterable of str (optional)
        Column names, if not already provided via ``X``.
    name : str
        Name of the matrix.
    prior_mu : array
        Prior means of the coefficients associated with each column in a linear
        regression problem.
    prior_sigma : array
        Prior standard deviations of the coefficients associated with each
        column in a linear regression problem.
    """
    def __init__(self, input, columns=None, name='unnamed_matrix', prior_mu=None,
                 prior_sigma=None, sparse=False):

        self.name = name
        self._sparse = sparse

        if issparse(input):
            # Build a sparse DM
            self._sparse = True
            self.X = input
            self.columns = columns
            self.df = None

        else:
            if not isinstance(input, pd.DataFrame):
                X = pd.DataFrame(input)
            else:
                X = deepcopy(input)
            self.df = X

            if columns is not None:
                self.columns = columns
            else:
                if hasattr(X, 'columns'):
                    self.columns = X.columns

            if self._sparse:
                self.X = lil_matrix(X.values)
            else:
                self.X = X.values

        if prior_mu is None:
            prior_mu = np.zeros(self.X.shape[1])
        if prior_sigma is None:
            prior_sigma = np.ones(self.X.shape[1]) * np.inf
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma


    def plot(self, ax=None, **kwargs):
        """Visualize the design matrix values as an image.

        Uses Matplotlib's `~lightkurve.utils.plot_image` to visualize the
        matrix values.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        **kwargs : dict
            Extra parameters to be passed to `.plot_image`.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        with plt.style.context(MPLSTYLE):
            values = self.values
            ax = plot_image(values, ax=ax, xlabel='Component', ylabel='X',
                            clabel='Component Value', title=self.name, **kwargs)
            ax.set_aspect(self.shape[1]/(1.6*self.shape[0]))
            if self.shape[1] <= 40:
                ax.set_xticks(np.arange(self.shape[1]))
                if self.columns is not None:
                    ax.set_xticklabels([r'${}$'.format(i) for i in self.columns],
                                       rotation=90, fontsize=8)
        return ax

    def plot_priors(self, ax=None):
        """Visualize the coefficient priors.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        def gauss(x, mu=0, sigma=1):
            return np.exp(-(x - mu)**2/(2*sigma**2))
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            for m, s in zip(self.prior_mu, self.prior_sigma):
                if ~np.isfinite(s):
                    ax.axhline(1, color='k')
                else:
                    x = np.linspace(m - 5*s, m + 5*s, 1000)
                    ax.plot(x, gauss(x, m, s), c='k')
            ax.set_xlabel('Value')
            ax.set_title('{} Priors'.format(self.name))
        return ax

    def _get_prior_sample(self):
        """Returns a random sample from the prior distribution."""
        return np.random.normal(self.prior_mu, self.prior_sigma)

    def split(self, row_indices, inplace=False):
        """Returns a new `.DesignMatrix` with regressors split into multiple
        columns.

        This method will return a new design matrix containing
        n_columns * len(row_indices) regressors.  This is useful in situations
        where the linear regression can be improved by fitting separate
        coefficients for different contiguous parts of the regressors.

        Parameters
        ----------
        row_indices : iterable of integers
            Every regressor (i.e. column) in the design matrix will be split
            up over multiple columns separated at the indices provided.

        Returns
        -------
        `.DesignMatrix`
            A new design matrix with shape (n_rows, len(row_indices)*n_columns).
        """
        if not hasattr(row_indices, '__iter__'):
            row_indices = list(row_indices)
        if len(row_indices) == 0:
            return self
        if (row_indices == [0]) or (row_indices == [self.shape[0]]):
            return self

        if inplace:
            dm = self
        else:
            dm = self.copy()

        x = np.arange(dm.shape[0])
        dm.prior_mu = np.concatenate([list(self.prior_mu) * (len(row_indices) + 1)])
        dm.prior_sigma = np.concatenate([list(self.prior_sigma) * (len(row_indices) + 1)])
        if isinstance(dm.df, pd.DataFrame):
            dm.df = pd.concat([((dm.df * np.atleast_2d(np.in1d(x, idx).astype(int)).T)).add_suffix('_{}'.format(jdx)) for jdx, idx in enumerate(np.array_split(x, row_indices))], axis=1)
            non_zero = dm.df.sum(axis=0) != 0
            dm.df = dm.df[dm.df.columns[non_zero]]
            dm.df = dm.df.values
            if not self._sparse:
                dm.X = dm.df.values
        if issparse(dm.X):
            dm.X = hstack([dm.X.multiply(lil_matrix(np.in1d(x, idx).astype(int)).T) for idx in np.array_split(x, row_indices)], format='lil')
            non_zero = dm.X.sum(axis=0) != 0
            non_zero = np.asarray(non_zero).ravel()
            dm.X = dm.X[:, non_zero]

        dm.prior_mu = dm.prior_mu[non_zero]
        dm.prior_sigma = dm.prior_sigma[non_zero]
        return dm

    def copy(self):
        """Returns a deepcopy of DesignMatrix"""
        return deepcopy(self)

    def standardize(self, inplace=False):
        """Returns a new `.DesignMatrix` in which the columns have been
        median-subtracted and sigma-divided.

        For each column in the matrix, this method will subtract the median of
        the column and divide by the column's standard deviation, i.e. it
        will compute the column's so-called "standard scores" or "z-values".

        This operation is useful because it will make the matrix easier to
        visualize and makes fitted coefficients easier to interpret.

        Notes:
        * Standardizing a spline design matrix will break the splines.
        * Columns with constant values (i.e. zero standard deviation) will be
        left unchanged.

        Returns
        -------
        `.DesignMatrix`
            A new design matrix with median-subtracted & sigma-divided columns.
        """

        if inplace:
            dm = self
        else:
            dm = self.copy()

        if isinstance(dm.df, pd.DataFrame):
            df_nan = dm.df.replace(0, np.nan)
            mean = df_nan.mean()
            std = df_nan.std()
            mean[std == 0] = 0
            std[std == 0] = 1
            dm.df = ((df_nan - mean) / std).fillna(0)
            dm.X = dm.df.values

        if issparse(dm.X):
            idx, jdx, v = find(dm.X)
            weights = dm.X.copy()
            weights[dm.X != 0] = 1
            mean = np.bincount(jdx, weights=v)/np.bincount(jdx)
            std = np.asarray([((np.sum((v[jdx == i] - mean[i])**2) * (1/((jdx == i).sum() - 1))))**0.5 for i in np.unique(jdx)])
            mean[std == 0] = 0
            std[std == 0] = 1
            white = (dm.X - vstack([lil_matrix(mean)] * dm.shape[0])).multiply(vstack([lil_matrix(1/std)] * dm.shape[0]))
            dm.X = white.multiply(weights)
        return dm


    def pca(self, nterms=6):
        """Returns a new `.DesignMatrix` with a smaller number of regressors.

        This method will use Principal Components Analysis (PCA) to reduce
        the number of columns in the matrix.

        Parameters
        ----------
        nterms : int
            Number of columns in the new matrix.

        Returns
        -------
        `.DesignMatrix`
            A new design matrix with PCA applied.
        """
        # nterms cannot be langer than the number of columns in the matrix
        if nterms > self.shape[1]:
            nterms = self.shape[1]
        # We use `fbpca.pca` instead of `np.linalg.svd` because it is faster.
        # Note that fbpca is randomized, and has n_iter=2 as default,
        # we find this to be too few, and that n_iter=10 is still fast but
        # produces more stable results.
        from fbpca import pca  # local import because not used elsewhere
        new_values, _, _ = pca(self.values, nterms, n_iter=10)
        return DesignMatrix(new_values, name=self.name, sparse=self._sparse)

    def append_constant(self, prior_mu=0, prior_sigma=np.inf, inplace=False):
        """Returns a new `.DesignMatrix` with a column of ones appended.

        Returns
        -------
        `.DesignMatrix`
            New design matrix with a column of ones appended. This column is
            named "offset".
        """
        if inplace:
            dm = self
        else:
            dm = self.copy()
        if isinstance(dm.X, pd.DataFrame):
            dm.df.insert(dm.shape[1], 'offset', 1, allow_duplicates=False)
            dm.X = dm.df.values
        else:
            dm.X = hstack([dm.X, lil_matrix(np.ones(dm.shape[0])).T], format='lil')
        dm.prior_mu = np.append(dm.prior_mu, prior_mu)
        dm.prior_sigma = np.append(dm.prior_sigma, prior_sigma)
        return dm

    def _validate(self):
        """Raises a `LightkurveWarning` if the matrix has a low rank."""
        # Matrix rank shouldn't be significantly smaller than the # of columns
        if self.rank < (0.5*self.shape[1]):
            warnings.warn("The design matrix has low rank ({}) compared to the "
                          "number of columns ({}), which suggests that the "
                          "matrix contains duplicate or correlated columns. "
                          "This may prevent the regression from succeeding. "
                          "Consider reducing the dimensionality by calling the "
                          "`pca()` method.".format(self.rank, self.shape[1]),
                          LightkurveWarning)

    @property
    def rank(self):
        """Matrix rank computed using `numpy.linalg.matrix_rank`."""
        return np.linalg.matrix_rank(self.values)

    @property
    def shape(self):
        """Tuple specifying the shape of the matrix as (n_rows, n_columns)."""
        return self.X.shape

    @property
    def values(self):
        """2D numpy array containing the matrix values."""
        if issparse(self.X):
            return self.X.toarray()
        return self.X

    # def __getitem__(self, key):
    #     return self.df[key]

    def __repr__(self):
        if self._sparse:
                return '{} DesignMatrix [sparse] {}'.format(self.name, self.shape)
        return '{} DesignMatrix {}'.format(self.name, self.shape)


class DesignMatrixCollection():
    """A set of design matrices."""
    def __init__(self, matrices):
        if np.all([issparse(m.X) for m in matrices]):
            self._sparse = True
#            self._sparse_values = hstack([m.values for m in matrices], format='lil')
        else:
            self._sparse = False
        self.matrices = matrices
        if self._sparse:
             self.X = hstack([m.X for m in matrices], format='csr')
        else:
             self.X = np.hstack([m.X for m in matrices])

    @property
    def values(self):
        """2D numpy array containing the matrix values."""
        return np.hstack(tuple(m.values for m in self.matrices))

    @property
    def df(self):
        """Stack of X as either np.array or sparse array"""
        if np.all([d.df is not None for d in self]):
             return pd.concat([m.df for m in self.matrices], axis=1)
        else:
            return None


    @property
    def prior_mu(self):
        """Coefficient prior means."""
        return np.hstack([m.prior_mu for m in self])

    @property
    def prior_sigma(self):
        """Coefficient prior standard deviations."""
        return np.hstack([m.prior_sigma for m in self])

    def plot(self, ax=None, **kwargs):
        """Visualize the design matrix values as an image.

        Uses Matplotlib's `~lightkurve.utils.plot_image` to visualize the
        matrix values.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        **kwargs : dict
            Extra parameters to be passed to `.plot_image`.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        temp_dm = self.flatten()
        ax = temp_dm.plot(**kwargs)
        ax.set_title("Design Matrix Collection")
        return ax

    def plot_priors(self, ax=None):
        """Visualize the `prior_mu` and `prior_sigma` attributes.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        [dm.plot_priors(ax=ax) for dm in self]
        return ax

    def _get_prior_sample(self):
        """Returns a random sample from the prior distribution."""
        return np.hstack([dm.sample_priors() for dm in self])

    def split(self, row_indices):
        """Returns a new `.DesignMatrixCollection` with regressors split into
        multiple columns.

        This method will return a new design matrix collection by calling
        `DesignMatrix.split` on each matrix in the collection.

        Parameters
        ----------
        row_indices : iterable of integers
            Every regressor (i.e. column) in the design matrix will be split
            up over multiple columns separated at the indices provided.

        Returns
        -------
        `.DesignMatrixCollection`
            A new design matrix collection.
        """
        return DesignMatrixCollection([d.split(row_indices) for d in self])

    def standardize(self):
        """Returns a new `.DesignMatrixCollection` in which all the
        matrices have been standardized using the `DesignMatrix.standardize`
        method.

        Returns
        -------
        `.DesignMatrixCollection`
            The new design matrix collection.
        """
        return DesignMatrixCollection([d.standardize() for d in self])

    @property
    def columns(self):
        """List of column names."""
        if np.all([d.columns is not None for d in self]):
            return np.hstack([d.columns for d in self])
        return None

    def flatten(self, name=None):
        """Flatten a DesignMatrixCollection into a DesignMatrix"""
        if name is None:
            name = self.matrices[0].name
        return DesignMatrix(self.X, columns=self.columns, prior_mu=self.prior_mu, prior_sigma=self.prior_sigma,
                            name=name, sparse=self._sparse)

    def __getitem__(self, key):
        try:
            return self.matrices[key]
        except Exception:
            arg = np.argwhere([m.name == key for m in self.matrices])
            return self.matrices[arg[0][0]]

    def _validate(self):
        [d._validate() for d in self]

    def __repr__(self):
        if self._sparse:
            return 'DesignMatrixCollection [sparse]:\n' + \
                        ''.join(['\t{}\n'.format(i.__repr__()) for i in self])
        return 'DesignMatrixCollection:\n' + \
                    ''.join(['\t{}\n'.format(i.__repr__()) for i in self])


####################################################
# Functions to create commonly-used design matrices.
####################################################

def create_spline_matrix(x, n_knots=20, knots=None, degree=3, name='spline',
                         include_intercept=True):
    """Returns a `.DesignMatrix` which models splines using `patsy.dmatrix`.

    Parameters
    ----------
    x : np.ndarray
        vector to spline
    n_knots: int
        Number of knots (default: 20).
    degree: int
        Polynomial degree.
    name: string
        Name to pass to `.DesignMatrix` (default: 'spline').
    include_intercept: bool
        Whether to include row of ones to find intercept. Default False.

    Returns
    -------
    dm: `.DesignMatrix`
        Design matrix object with shape (len(x), n_knots*degree).
    """
    from patsy import dmatrix  # local import because it's rarely-used
    if knots is not None:
        dm_formula = "bs(x, knots={}, degree={}, include_intercept={}) - 1" \
                     "".format(knots, degree, include_intercept)
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
        df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
                                              for idx in range(spline_dm.shape[1])])
    else:
        dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" \
                 "".format(n_knots, degree, include_intercept)
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
        df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
                                              for idx in range(n_knots)])
    return DesignMatrix(df, name=name)




@jit(nopython=True)
def basis(x, degree, i, knots):
    if degree == 0:
        B = np.zeros(len(x))
        B[(x >= knots[i]) & (x <= knots[i+1])] = 1
        #B = (B)
    else:
#        alpha1, alpha2 = 0, 0
        da = (knots[degree + i] - knots[i])
        db = (knots[i + degree + 1] - knots[i + 1])
        if ((knots[degree + i] - knots[i]) != 0):
            alpha1 = ((x - knots[i])/da)
        else:
            alpha1 = np.zeros(len(x))
        if ((knots[i+degree+1] - knots[i+1]) != 0):
            alpha2 = ((knots[i + degree + 1] - x)/db)
        else:
            alpha2 = np.zeros(len(x))
        B = (basis(x, (degree-1), i, knots)) * (alpha1) + (basis(x, (degree-1), (i+1), knots)) * (alpha2)
    return B




def create_sparse_spline_matrix(x, n_knots=20, knots=None, degree=3, name='spline'):
    """Returns a `.DesignMatrix` which models which are

    Parameters
    ----------
    x : np.ndarray
        vector to spline
    n_knots: int
        Number of knots (default: 20).
    knots : np.ndarray [optional]
        Optional array containing knots
    degree: int
        Polynomial degree.
    name: string
        Name to pass to `.DesignMatrix` (default: 'spline').
    include_intercept: bool
        Whether to include row of ones to find intercept. Default False.

    Returns
    -------
    dm: `.DesignMatrix`
        Design matrix object with shape (len(x), n_knots*degree).
    """

    # To use jit we have to use float64
    x = np.asarray(x, np.float64)


    if not isinstance(n_knots, int):
        raise ValueError('`n_knots` must be an integer.')
    if n_knots - degree <= 0:
        raise ValueError('n_knots must be greater than degree.')
    if (knots is None)  and (n_knots is not None):
        knots = np.asarray([s[-1] for s in np.array_split(np.argsort(x), n_knots - degree)[:-1]])
        knots = [np.mean([x[k], x[k + 1]]) for k in knots]
        knots = np.append(np.append(x.min(), knots), x.max())
#        knots = np.linspace(x.min(), x.max(), n_knots - 2)
    elif (knots is None)  and (n_knots is None):
        raise ValueError('Pass either `n_knots` or `knots`.')
    knots_wbounds = np.append(np.append([x.min()] * (degree - 1), knots), [x.max()] * (degree))

    matrices = [csr_matrix(basis(x, degree, idx, knots_wbounds)) for idx in np.arange(-1, len(knots_wbounds) - degree - 1)]
    spline_dm = vstack([m for m in matrices if (m.sum() != 0) ], format='csr').T
    return DesignMatrix(spline_dm, name=name)
    # df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
    #                                       for idx in range(n_knots)])
    # return DesignMatrix(df, name=name, sparse=sparse)
