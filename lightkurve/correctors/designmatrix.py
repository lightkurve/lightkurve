"""Defines `DesignMatrix` and `DesignMatrixCollection`.

These classes are intended to make linear regression problems with a large
design matrix more easy.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    df : dict, array, or `pandas.DataFrame` object
        Columns to include in the design matrix.  If this object is not a
        `~pandas.DataFrame` then it will be passed to the DataFrame constructor.
    columns : iterable of str (optional)
        Column names, if not already provided via ``df``.
    name : str
        Name of the matrix.
    prior_mu : array
        Prior means of the coefficients associated with each column in a linear
        regression problem.
    prior_sigma : array
        Prior standard deviations of the coefficients associated with each
        column in a linear regression problem.
    """
    def __init__(self, df, columns=None, name='unnamed_matrix', prior_mu=None,
                 prior_sigma=None):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if columns is not None:
            df.columns = columns
        self.df = df
        self.name = name
        if prior_mu is None:
            prior_mu = np.zeros(len(df.T))
        if prior_sigma is None:
            prior_sigma = np.ones(len(df.T)) * np.inf
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
            ax = plot_image(self.values, ax=ax, xlabel='Component', ylabel='X',
                            clabel='Component Value', title=self.name, **kwargs)
            ax.set_aspect(self.shape[1]/(1.6*self.shape[0]))
            if self.shape[1] <= 40:
                ax.set_xticks(np.arange(self.shape[1]))
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

    def split(self, row_indices):
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
        if isinstance(row_indices, int):
            row_indices = [row_indices]
        if (len(row_indices) == 0) or (row_indices == [0]) or (row_indices is None):
            return self
        # Where do the submatrices begin and end?
        lower_idx = np.append(0, row_indices)
        upper_idx = np.append(row_indices, len(self.df))
        dfs = []
        for idx, a, b in zip(range(len(lower_idx)), lower_idx, upper_idx):
            new_columns = dict(
                ('{}'.format(val), '{}'.format(val) + ' {}'.format(idx + 1))
                for val in list(self.df.columns))
            dfs.append(self.df[a:b].rename(columns=new_columns))
        new_df = pd.concat(dfs, axis=1).fillna(0)
        prior_mu = np.hstack([self.prior_mu for idx in range(len(dfs))])
        prior_sigma = np.hstack([self.prior_sigma for idx in range(len(dfs))])
        return DesignMatrix(new_df, name=self.name, prior_mu=prior_mu,
                            prior_sigma=prior_sigma)

    def standardize(self):
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
        ar = np.asarray(np.copy(self.df))
        ar[ar == 0] = np.nan
        # If a column has zero standard deviation, it will not change!
        is_const = np.nanstd(ar, axis=0) == 0
        median = np.atleast_2d(np.nanmedian(ar, axis=0)[~is_const])
        std = np.atleast_2d(np.nanstd(ar, axis=0)[~is_const])
        ar[:, ~is_const] = (ar[:, ~is_const] - median) / std
        new_df = pd.DataFrame(ar, columns=self.columns).fillna(0)
        return DesignMatrix(new_df, name=self.name)

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
        return DesignMatrix(new_values, name=self.name)

    def append_constant(self, prior_mu=0, prior_sigma=np.inf):
        """Returns a new `.DesignMatrix` with a column of ones appended.

        Returns
        -------
        `.DesignMatrix`
            New design matrix with a column of ones appended. This column is
            named "offset".
        """
        extra_df = pd.DataFrame(np.atleast_2d(np.ones(self.shape[0])).T, columns=['offset'])
        new_df = pd.concat([self.df, extra_df], axis=1)
        prior_mu = np.append(self.prior_mu, prior_mu)
        prior_sigma = np.append(self.prior_sigma, prior_sigma)
        return DesignMatrix(new_df, name=self.name,
                            prior_mu=prior_mu, prior_sigma=prior_sigma)

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
    def columns(self):
        """List of column names."""
        return list(self.df.columns)

    @property
    def shape(self):
        """Tuple specifying the shape of the matrix as (n_rows, n_columns)."""
        return self.df.shape

    @property
    def values(self):
        """2D numpy array containing the matrix values."""
        return self.df.values

    def __getitem__(self, key):
        return self.df[key]

    def __repr__(self):
        return '{} DesignMatrix {}'.format(self.name, self.shape)


class DesignMatrixCollection():
    """A set of design matrices."""
    def __init__(self, matrices):
        self.matrices = matrices

    @property
    def values(self):
        """2D numpy array containing the matrix values."""
        return np.hstack(tuple(m.values for m in self.matrices))

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
        temp_dm = DesignMatrix(pd.concat([d.df for d in self], axis=1))
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
        return np.hstack([d.columns for d in self])

    def __getitem__(self, key):
        try:
            return self.matrices[key]
        except Exception:
            arg = np.argwhere([m.name == key for m in self.matrices])
            return self.matrices[arg[0][0]]

    def _validate(self):
        [d._validate() for d in self]

    def __repr__(self):
        return 'DesignMatrixCollection:\n' + \
                    ''.join(['\t{}\n'.format(i.__repr__()) for i in self])


####################################################
# Functions to create commonly-used design matrices.
####################################################

def create_spline_matrix(x, n_knots=20, degree=3, name='spline',
                         include_intercept=False):
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
    dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" \
                 "".format(n_knots, degree, include_intercept)
    spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
    df = pd.DataFrame(spline_dm, columns=['knot{}'.format(idx + 1)
                                          for idx in range(n_knots)])
    return DesignMatrix(df, name=name)
