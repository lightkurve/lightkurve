"""Defines `DesignMatrix` and `DesignMatrixCollection`.

TODO
----
* Improve user input validation and error checking.
* Add a warning if the column rank of the matrix is bad, i.e. if the matrix has
  tightly-correlated regressors?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import MPLSTYLE


__all__ = ['DesignMatrix', 'DesignMatrixCollection']

class DesignMatrixException(Exception):
    pass

class DesignMatrix():
    """A matrix of column vectors for use in linear regression.

    The purpose of this class is to provide a convenient method to interact
    with a set of one or more regressors which are known to correlate with
    trends or systematic noise signals which we wants to remove from a light
    curve. Specifically, this class is designed to provide the regressors
    to Lightkurve's `~lightkurve.corrector.RegressionCorrector` class.

    Parameters
    ----------
    df : pandas `DataFrame` object
    columns : iterable of str (optional)
    """
    def __init__(self, df, columns=None, name='unnamed_matrix', prior_mu=None, prior_sigma=None):
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

    def plot(self, ax=None, show_colorbar=True, **kwargs):
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            im = ax.imshow(self.values, origin='bottom', **kwargs)
            ax.set_aspect(self.shape[1]/(1.6*self.shape[0]))
            ax.set_xlabel('Component')
            ax.set_ylabel('X')
            ax.set_title(self.name)
            if self.shape[1] <= 40:
                ax.set_xticks(np.arange(self.shape[1]))
                ax.set_xticklabels([r'${}$'.format(i) for i in self.columns], rotation=90, fontsize=8)
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Component Value')
        return ax

    def plot_priors(self, ax=None):
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

    def sample_priors(self):
        return np.random.normal(self.prior_mu, self.prior_sigma)

    def split(self, row_indices):
        """Returns a new matrix with the regressors split over multiple columns.

        This method will return a new design matrix containing
        n_columns*len(row_indices) regressors.  This is useful in situations
        where the linear regression can be improved by fitting separate
        coefficients for different contiguous parts of the regressors.

        Parameters
        ----------
        row_indices : iterable of integers
            Every regressor (i.e. column) in the design matrix will be split
            up over multiple columns separated at the indices provided.

        Returns
        -------
        `~lightkurve.correctors.DesignMatrix`
            A new design matrix with shape (n_rows, len(row_indices)*n_columns).
        """

        if isinstance(row_indices, int):
            row_indices = [row_indices]
        if (len(row_indices) == 0) | (row_indices == [0]) | (row_indices == None) :
            return self
        # Where do the submatrices begin and end?
        lower_idx = np.append(0, row_indices)
        upper_idx = np.append(row_indices, len(self.df))
        dfs = []
        for idx, a, b in zip(range(len(lower_idx)), lower_idx, upper_idx):
            new_columns = dict(('{}'.format(val), '{}'.format(val) + ' {}'.format(idx + 1))
                                for val in list(self.df.columns))
            dfs.append(self.df[a:b].rename(columns=new_columns))
        new_df = pd.concat(dfs, axis=1).fillna(0)
        prior_mu = np.hstack([self.prior_mu for idx in range(len(dfs))])
        prior_sigma = np.hstack([self.prior_sigma for idx in range(len(dfs))])
        return DesignMatrix(new_df, name=self.name, prior_mu=prior_mu, prior_sigma=prior_sigma)

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
        `~lightkurve.correctors.DesignMatrix`
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
        `~lightkurve.correctors.DesignMatrix`
            A new design matrix with PCA applied.
        """
        # nterms cannot be langer than the number of columns in the matrix
        if nterms > self.shape[1]:
            nterms = self.shape[1]
        # `fbpca.pca` is faster than `np.linalg.svd` but an optional dependency
        try:
            from fbpca import pca
            # fbpca is randomized, and has n_iter=2 as default,
            # we find this to be too few, and that n_iter=10 is still fast but
            # produces stable results.
            new_values, _, _ = pca(self.values, nterms, n_iter=10)
        except (ImportError, ModuleNotFoundError):
            new_values, _, _ = np.linalg.svd(self.values)
            new_values = new_values[:, :nterms]
        return DesignMatrix(new_values, name=self.name)

    def append_constant(self, prior_mu=0, prior_sigma=np.inf):
        new_df = pd.concat([self.df, pd.DataFrame(np.atleast_2d(np.ones(self.shape[0])).T, columns=['offset'])], axis=1)
        prior_mu = np.append(self.prior_mu, prior_mu)
        prior_sigma = np.append(self.prior_sigma, prior_sigma)

        return DesignMatrix(new_df, name=self.name, prior_mu=prior_mu, prior_sigma=prior_sigma)

    def _validate(self):
        """Raises a `DesignMatrixException` if the matrix contains identical columns."""
        # for idx in range(self.shape[1]):
        #     dupes = np.any([np.allclose(self.values[:, idx], self.values[:, jdx])
        #                     for jdx in np.arange(idx + 1, self.shape[1])])
        #     if dupes:
        #         raise DesignMatrixException("Design Matrix contains duplicate columns.")
        self.rank = np.linalg.matrix_rank(self.values)
        if self.shape[1] < (self.rank - 1):
            warnings.warn('Matrix has low rank, matrix might contain duplicate columns', lk.LightkurveWarning)

    @property
    def columns(self):
        return list(self.df.columns)

    @property
    def shape(self):
        return self.df.shape

    @property
    def values(self):
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
        return np.hstack(tuple(m.values for m in self.matrices))

    @property
    def prior_mu(self):
        return np.hstack([m.prior_mu for m in self])

    @property
    def prior_sigma(self):
        return np.hstack([m.prior_sigma for m in self])

    def plot(self, ax=None, **kwargs):
        temp_dm = DesignMatrix(pd.concat([d.df for d in self], axis=1))
        ax = temp_dm.plot(**kwargs)
        ax.set_title("Design Matrix Collection")
        return ax

    def plot_priors(self, ax=None):
        [dm.plot_priors() for dm in self]
        return

    def sample_priors(self):
        return np.hstack([dm.sample_priors() for dm in self])

    def split(self, row_indices):
        return DesignMatrixCollection([d.split(row_indices) for d in self])

    def standardize(self):
        return DesignMatrixCollection([d.standardize() for d in self])

    @property
    def columns(self):
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
        return 'DesignMatrixCollection:\n' + ''.join(['\t{}\n'.format(i.__repr__()) for i in self])
