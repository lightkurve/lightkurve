"""Defines design matrix objects to aid linear regression problems.

Specifically, this module adds the `DesignMatrix`, `DesignMatrixCollection`,
`SparseDesignMatrix`, and `SparseDesignMatrixCollection` classes which
are design to work with the `RegressionCorrector` class.
"""
from copy import deepcopy
import warnings

from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix, hstack, vstack, issparse, find

from .. import MPLSTYLE
from ..utils import LightkurveWarning, plot_image


__all__ = [
    "DesignMatrix",
    "SparseDesignMatrix",
    "DesignMatrixCollection",
    "SparseDesignMatrixCollection",
]


class DesignMatrix:
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

    Examples
    --------
    >>> from lightkurve.correctors.designmatrix import DesignMatrix, create_spline_matrix
    >>> DesignMatrix(np.arange(100), name='slope')
    slope DesignMatrix (100, 1)
    >>> create_spline_matrix(np.arange(100), n_knots=5, name='spline')
    spline DesignMatrix (100, 5)
    """

    def __init__(
        self, df, columns=None, name="unnamed_matrix", prior_mu=None, prior_sigma=None
    ):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        self.df = df
        if columns is not None:
            df.columns = columns
        self.columns = list(df.columns)
        self.name = name

        if isinstance(prior_mu, u.Quantity):
            prior_mu = prior_mu.value
        if prior_mu is None:
            prior_mu = np.zeros(len(df.T))
        self.prior_mu = np.atleast_1d(prior_mu)

        if isinstance(prior_sigma, u.Quantity):
            prior_sigma = prior_sigma.value
        if prior_sigma is None:
            prior_sigma = np.ones(len(df.T)) * np.inf
        self.prior_sigma = np.atleast_1d(prior_sigma)

    @property
    def X(self):
        """Design matrix "X" to be used in RegressionCorrector objects"""
        return self.df.values

    def copy(self):
        """Returns a deepcopy of DesignMatrix"""
        return deepcopy(self)

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
            ax = plot_image(
                self.values,
                ax=ax,
                xlabel="Component",
                ylabel="X",
                clabel="Component Value",
                title=self.name,
                interpolation="nearest",
                **kwargs
            )
            ax.set_aspect(self.shape[1] / (1.6 * self.shape[0]))
            if self.shape[1] <= 40:
                ax.set_xticks(np.arange(self.shape[1]))
                ax.set_xticklabels(
                    [r"${}$".format(i) for i in self.columns], rotation=90, fontsize=8
                )
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
            return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            for m, s in zip(self.prior_mu, self.prior_sigma):
                if ~np.isfinite(s):
                    ax.axhline(1, color="k")
                else:
                    x = np.linspace(m - 5 * s, m + 5 * s, 1000)
                    ax.plot(x, gauss(x, m, s), c="k")
            ax.set_xlabel("Value")
            ax.set_title("{} Priors".format(self.name))
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
                ("{}".format(val), "{}".format(val) + " {}".format(idx + 1))
                for val in list(self.df.columns)
            )
            dfs.append(self.df[a:b].rename(columns=new_columns))

        prior_mu = np.hstack([self.prior_mu for idx in range(len(dfs))])
        prior_sigma = np.hstack([self.prior_sigma for idx in range(len(dfs))])

        if inplace:
            dm = self
        else:
            dm = self.copy()
        dm.df = pd.concat(dfs, axis=1).fillna(0)
        dm.columns = dm.df.columns
        dm.prior_mu = prior_mu
        dm.prior_sigma = prior_sigma
        return dm

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
        ar = np.asarray(np.copy(self.df))
        ar[ar == 0] = np.nan
        # If a column has zero standard deviation, it will not change!
        is_const = np.nanstd(ar, axis=0) == 0
        median = np.atleast_2d(np.nanmedian(ar, axis=0)[~is_const])
        std = np.atleast_2d(np.nanstd(ar, axis=0)[~is_const])
        ar[:, ~is_const] = (ar[:, ~is_const] - median) / std
        new_df = pd.DataFrame(ar, columns=self.columns).fillna(0)
        if inplace:
            dm = self
        else:
            dm = self.copy()
        dm.df = new_df
        return dm

    def pca(self, nterms=6, n_iter=10):
        """Returns a new `.DesignMatrix` with a smaller number of regressors.

        This method will use Principal Components Analysis (PCA) to reduce
        the number of columns in the matrix.

        Parameters
        ----------
        nterms : int
            Number of columns in the new matrix.

        n_iter : int
            Number of iterations that will be run by the power iteration
            algorithm to compute the principal components.

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

        new_values, _, _ = pca(self.values, nterms, n_iter=n_iter)
        return DesignMatrix(new_values, name=self.name)

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
        extra_df = pd.DataFrame(
            np.atleast_2d(np.ones(self.shape[0])).T, columns=["offset"]
        )
        dm.df = pd.concat([self.df, extra_df], axis=1)
        dm.columns = list(dm.df.columns)
        dm.prior_mu = np.append(self.prior_mu, prior_mu)
        dm.prior_sigma = np.append(self.prior_sigma, prior_sigma)
        return dm

    def _validate(self, rank=True):
        """Helper function for validating."""
        # Matrix rank shouldn't be significantly smaller than the # of columns
        if rank:
            if self.rank < (0.5 * self.shape[1]):
                warnings.warn(
                    "The design matrix has low rank ({}) compared to the "
                    "number of columns ({}), which suggests that the "
                    "matrix contains duplicate or correlated columns. "
                    "This may prevent the regression from succeeding. "
                    "Consider reducing the dimensionality by calling the "
                    "`pca()` method.".format(self.rank, self.shape[1]),
                    LightkurveWarning,
                )
        if self.prior_mu is not None:
            if len(self.prior_mu) != self.shape[1]:
                raise ValueError(
                    "`prior_mu` must have shape {}" "".format(self.shape[1])
                )
        if self.prior_sigma is not None:
            if len(self.prior_sigma) != self.shape[1]:
                raise ValueError(
                    "`prior_sigma` must have shape {}" "".format(self.shape[1])
                )
            if np.any(np.asarray(self.prior_sigma) <= 0):
                raise ValueError(
                    "`prior_sigma` values cannot be smaller than " "or equal to zero"
                )

    def validate(self, rank=True):
        """Emits `LightkurveWarning` if matrix has low rank or priors have incorrect shape.

        Note that for `SparseDesignMatrix` objects, calculating the rank will
        force the design matrix to be evaluated and stored in memory, reducing
        the speed and memory savings of SparseDesignMatrix.

        For `SparseDesignMatrix`, rank checks will be turned off by default.
        """
        self._validate()

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
        return self.df.values

    def __getitem__(self, key):
        return self.df[key].values

    def __repr__(self):
        return "{} DesignMatrix {}".format(self.name, self.shape)

    def to_sparse(self):
        """Convert this dense matrix object to a `SparseDesignMatrix`.

        The values of this design matrix will be converted to a
        `scipy.sparse.csr_matrix`, which stores the values in a
        lower memory matrix. This is not recommended for dense matrices.
        """
        return SparseDesignMatrix(
            csr_matrix(self.values),
            name=self.name,
            columns=self.columns,
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
        )

    def collect(self, matrix):
        """ Join two designmatrices, return a design matrix collection """
        return DesignMatrixCollection([self, matrix])


class DesignMatrixCollection:
    """Object which stores multiple design matrices.

    DesignMatrixCollection objects are useful when users want to regress against
    multiple different systematics, but still keep the different systematics distinct.

    Examples
    --------
    >>> from lightkurve.correctors.designmatrix import create_spline_matrix, DesignMatrix, DesignMatrixCollection
    >>> dm1 = create_spline_matrix(np.arange(100), n_knots=5, name='spline')
    >>> dm2 = DesignMatrix(np.arange(100), name='slope')
    >>> dmc = DesignMatrixCollection([dm1, dm2])
    >>> dmc
    DesignMatrixCollection:
        spline DesignMatrix (100, 5)
        slope DesignMatrix (100, 1)
    >>> dmc.matrices
    [spline DesignMatrix (100, 5), slope DesignMatrix (100, 1)]
    """

    def __init__(self, matrices):
        if np.any([issparse(m.X) for m in matrices]):
            # This collection is designed for dense matrices, so we warn if a
            # SparseDesignMatrix is passed
            warnings.warn(
                (
                    "Some matrices are `SparseDesignMatrix` objects. "
                    "Sparse matrices will be converted to dense matrices."
                ),
                LightkurveWarning,
            )
            dense_matrices = []
            for m in matrices:
                if isinstance(m, SparseDesignMatrix):
                    dense_matrices.append(m.copy().to_dense())
                else:
                    dense_matrices.append(m)
            self.matrices = dense_matrices
        else:
            self.matrices = matrices
        self.X = np.hstack(tuple(m.X for m in self.matrices))
        self._child_class = DesignMatrix
        self.validate()

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
        return self.__class__([d.split(row_indices) for d in self])

    def standardize(self):
        """Returns a new `.DesignMatrixCollection` in which all the
        matrices have been standardized using the `DesignMatrix.standardize`
        method.

        Returns
        -------
        `.DesignMatrixCollection`
            The new design matrix collection.
        """
        return self.__class__([d.standardize() for d in self])

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

    def validate(self):
        [d.validate() for d in self]

    def __repr__(self):
        return "DesignMatrixCollection:\n" + "".join(
            ["\t{}\n".format(i.__repr__()) for i in self]
        )

    def to_designmatrix(self, name=None):
        """Flatten a `DesignMatrixCollection` into a `DesignMatrix`."""
        if name is None:
            name = self.matrices[0].name
        return self._child_class(
            self.X,
            columns=self.columns,
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            name=name,
        )


class SparseDesignMatrix(DesignMatrix):
    """A matrix of column vectors for use in linear regression.

    This class is similar to the `DesignMatrix` class, but uses the
    `scipy.sparse` library to improve speed in the case of sparse matrices.

    The purpose of this class is to provide a convenient method to interact
    with a set of one or more regressors which are known to correlate with
    trends or systematic noise signals which we want to remove from a light
    curve. Specifically, this class is designed to provide the design matrix
    for use by Lightkurve's `.RegressionCorrector` class.

    Parameters
    ----------
    X : `scipy.sparse` matrix
        The values to build the design matrix with
    columns : iterable of str (optional)
        Column names
    name : str
        Name of the matrix.
    prior_mu : array
        Prior means of the coefficients associated with each column in a linear
        regression problem.
    prior_sigma : array
        Prior standard deviations of the coefficients associated with each
        column in a linear regression problem.
    """

    def __init__(
        self, X, columns=None, name="unnamed_matrix", prior_mu=None, prior_sigma=None
    ):
        if not issparse(X):
            raise ValueError(
                "Must pass a `scipy.sparse` matrix (e.g. `scipy.sparse.csr_matrix`)"
            )
        if columns is None:
            columns = np.arange(X.shape[1])
        self.columns = columns
        self.name = name
        if prior_mu is None:
            prior_mu = np.zeros(X.shape[1])
        if prior_sigma is None:
            prior_sigma = np.ones(X.shape[1]) * np.inf
        self._X = X
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self._child_class = SparseDesignMatrix
        self.validate()

    @property
    def X(self):
        """Design matrix "X" to be used in RegressionCorrector objects"""
        return self._X

    @property
    def values(self):
        """2D numpy array containing the matrix values."""
        return self.X.toarray()

    def validate(self, rank=False):
        """Checks if the matrix has the right shapes. Set rank to True to test matrix rank."""
        # For sparse matrices, calculating the rank is expensive, and negates
        # the benefits of using sparse. Validate will ignore rank by default.
        self._validate(rank=rank)

    def split(self, row_indices, inplace=False):
        """Returns a new `.SparseDesignMatrix` with regressors split into multiple
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
        `.SparseDesignMatrix`
            A new design matrix with shape (n_rows, len(row_indices)*n_columns).
        """

        if not hasattr(row_indices, "__iter__"):
            row_indices = [row_indices]

        # You can't split on the first or last index
        row_indices = list(
            np.asarray(row_indices)[~np.in1d(row_indices, [0, self.shape[0]])]
        )
        if len(row_indices) == 0:
            return self
        if inplace:
            dm = self
        else:
            dm = self.copy()
        x = np.arange(dm.shape[0])
        dm.prior_mu = np.concatenate([list(self.prior_mu) * (len(row_indices) + 1)])
        dm.prior_sigma = np.concatenate(
            [list(self.prior_sigma) * (len(row_indices) + 1)]
        )
        dm._X = hstack(
            [
                dm.X.multiply(lil_matrix(np.in1d(x, idx).astype(int)).T)
                for idx in np.array_split(x, row_indices)
            ],
            format="lil",
        )
        non_zero = dm.X.sum(axis=0) != 0
        non_zero = np.asarray(non_zero).ravel()
        dm._X = dm.X[:, non_zero]
        if dm.columns is not None:
            dm.columns = list(
                np.asarray(
                    [
                        ["{}_{}".format(c, idx) for c in dm.columns]
                        for idx in range(len(row_indices) + 1)
                    ]
                ).ravel()
            )
        dm.prior_mu = dm.prior_mu[non_zero]
        dm.prior_sigma = dm.prior_sigma[non_zero]
        return dm

    def standardize(self, inplace=False):
        """Returns a new `.SparseDesignMatrix` in which the columns have been
        mean-subtracted and sigma-divided.

        For each column in the matrix, this method will subtract the mean of
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
        `.SparseDesignMatrix`
            A new design matrix with mean-subtracted & sigma-divided columns.
        """
        if inplace:
            dm = self
        else:
            dm = self.copy()

        idx, jdx, v = find(dm.X)
        weights = dm.X.copy()
        weights[dm.X != 0] = 1
        mean = np.bincount(jdx, weights=v) / np.bincount(jdx)
        std = np.asarray(
            [
                ((np.sum((v[jdx == i] - mean[i]) ** 2) * (1 / ((jdx == i).sum() - 1))))
                ** 0.5
                for i in np.unique(jdx)
            ]
        )
        mean[std == 0] = 0
        std[std == 0] = 1
        white = (dm.X - vstack([lil_matrix(mean)] * dm.shape[0])).multiply(
            vstack([lil_matrix(1 / std)] * dm.shape[0])
        )
        dm._X = white.multiply(weights)
        return dm

    def pca(self, nterms=6, **kwargs):
        """Returns a new `.SparseDesignMatrix` with a smaller number of regressors.

        This method will use Principal Components Analysis (PCA) to reduce
        the number of columns in the matrix.

        Parameters
        ----------
        nterms : int
            Number of columns in the new matrix.

        Returns
        -------
        `.SparseDesignMatrix`
            A new design matrix with PCA applied.
        """
        return super().pca(nterms, **kwargs).to_sparse()

    def append_constant(self, prior_mu=0, prior_sigma=np.inf, inplace=False):
        """Returns a new `.SparseDesignMatrix` with a column of ones appended.

        Returns
        -------
        `.SparseDesignMatrix`
            New design matrix with a column of ones appended. This column is
            named "offset".
        """
        if inplace:
            dm = self
        else:
            dm = self.copy()
        dm._X = hstack([dm.X, lil_matrix(np.ones(dm.shape[0])).T], format="lil")
        dm.prior_mu = np.append(dm.prior_mu, prior_mu)
        dm.prior_sigma = np.append(dm.prior_sigma, prior_sigma)
        return dm

    def __getitem__(self, key):
        loc = np.where(np.asarray(self.columns) == key)[0]
        if len(loc) == 0:
            raise ValueError("No such column as `{}`.".format(key))
        return self.X[:, loc].toarray()

    def __repr__(self):
        return "{} SparseDesignMatrix {}".format(self.name, self.shape)

    def collect(self, matrix):
        """ Join two designmatrices, return a design matrix collection """
        return SparseDesignMatrixCollection([self, matrix])

    def to_dense(self):
        """Convert a SparseDesignMatrix object to a dense DesignMatrix

        The values of this design matrix will be converted to a
        `numpy.ndarray`. This is not recommended for sparse matrices containing
        mostly zeros.
        """
        return DesignMatrix(
            self.values,
            name=self.name,
            columns=self.columns,
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
        )


class SparseDesignMatrixCollection(DesignMatrixCollection):
    """A set of design matrices."""

    def __init__(self, matrices):
        if not np.all([issparse(m.X) for m in matrices]):
            # This collection is designed for sparse matrices, so we raise a warning if a dense DesignMatrix is passed
            warnings.warn(
                (
                    "Not all matrices are `SparseDesignMatrix` objects. "
                    "Dense matrices will be converted to sparse matrices."
                ),
                LightkurveWarning,
            )
            sparse_matrices = []
            for m in matrices:
                if isinstance(m, DesignMatrix):
                    sparse_matrices.append(m.copy().to_sparse())
                else:
                    sparse_matrices.append(m)
            self.matrices = sparse_matrices
        else:
            self.matrices = matrices
        self.X = hstack([m.X for m in self.matrices], format="csr")
        self._child_class = SparseDesignMatrix
        self.validate()

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
        temp_dm = SparseDesignMatrix(hstack([d.X for d in self]))
        ax = temp_dm.plot(**kwargs)
        ax.set_title("Design Matrix Collection")
        return ax

    def __repr__(self):
        return "SparseDesignMatrixCollection:\n" + "".join(
            ["\t{}\n".format(i.__repr__()) for i in self]
        )


####################################################
# Functions to create commonly-used design matrices.
####################################################

def _spline_basis_vector(x, degree, i, knots):
    """Recursive function to create a single spline basis vector for an input x,
    for the ith knot.

    See https://en.wikipedia.org/wiki/B-spline for a definition of B-spline
    basis vectors

    Parameters
    ----------
    x : np.ndarray
        Input x
    degree : int
        Degree of spline to calculate basis for
    i : int
        The index of the knot to calculate the basis for
    knots : np.ndarray
        Array of all knots

    Returns
    -------
    B : np.ndarray
        A vector of same length as x containing the spline basis for the ith knot
    """
    if degree == 0:
        B = np.zeros(len(x))
        B[(x >= knots[i]) & (x <= knots[i + 1])] = 1
    else:
        da = knots[degree + i] - knots[i]
        db = knots[i + degree + 1] - knots[i + 1]
        if (knots[degree + i] - knots[i]) != 0:
            alpha1 = (x - knots[i]) / da
        else:
            alpha1 = np.zeros(len(x))
        if (knots[i + degree + 1] - knots[i + 1]) != 0:
            alpha2 = (knots[i + degree + 1] - x) / db
        else:
            alpha2 = np.zeros(len(x))
        B = (_spline_basis_vector(x, (degree - 1), i, knots)) * (alpha1) + (
            _spline_basis_vector(x, (degree - 1), (i + 1), knots)
        ) * (alpha2)
    return B


def create_sparse_spline_matrix(x, n_knots=20, knots=None, degree=3, name="spline"):
    """Creates a piecewise polynomial function, creating a continuous, smooth function in x

    See https://en.wikipedia.org/wiki/B-spline for the definitions of Basis Splines

    B-spline vectors of degree higher than 0 are created using recursion, using the
    `_spline_basis_vector` function to evaluate the basis vectors for x, for each knot.

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
        Name to pass to `.SparseDesignMatrix` (default: 'spline').
    include_intercept: bool
        Whether to include row of ones to find intercept. Default False.

    Returns
    -------
    dm: `.SparseDesignMatrix`
        Design matrix object with shape (len(x), n_knots*degree).
    """
    # To use jit we have to use float64
    x = np.asarray(x, np.float64)

    if not isinstance(n_knots, int):
        raise ValueError("`n_knots` must be an integer.")
    if n_knots - degree <= 0:
        raise ValueError("n_knots must be greater than degree.")
    if (knots is None) and (n_knots is not None):
        knots = np.asarray(
            [s[-1] for s in np.array_split(np.argsort(x), n_knots - degree)[:-1]]
        )
        knots = [np.mean([x[k], x[k + 1]]) for k in knots]
    elif (knots is None) and (n_knots is None):
        raise ValueError("Pass either `n_knots` or `knots`.")
    knots = np.append(np.append(x.min(), knots), x.max())
    knots = np.unique(knots)
    knots_wbounds = np.append(
        np.append([x.min()] * (degree - 1), knots), [x.max()] * (degree)
    )

    matrices = [
        csr_matrix(_spline_basis_vector(x, degree, idx, knots_wbounds))
        for idx in np.arange(-1, len(knots_wbounds) - degree - 1)
    ]
    spline_dm = vstack([m for m in matrices if (m.sum() != 0)], format="csr").T
    return SparseDesignMatrix(spline_dm, name=name)


def create_spline_matrix(
    x, n_knots=20, knots=None, degree=3, name="spline", include_intercept=True
):
    """Returns a `.DesignMatrix` which models splines using `patsy.dmatrix`.

    Parameters
    ----------
    x : np.ndarray
        vector to spline
    n_knots: int
        Number of knots (default: 20).
    knots: list [optional]
        The interior knots to use for the spline. If unspecified, then equally
        spaced quantiles of the input data are used such that there are `n_knots` knots.
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
        dm_formula = "bs(x, knots={}, degree={}, include_intercept={}) - 1" "".format(
            knots, degree, include_intercept
        )
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
        df = pd.DataFrame(
            spline_dm,
            columns=["knot{}".format(idx + 1) for idx in range(spline_dm.shape[1])],
        )
    else:
        dm_formula = "bs(x, df={}, degree={}, include_intercept={}) - 1" "".format(
            n_knots, degree, include_intercept
        )
        spline_dm = np.asarray(dmatrix(dm_formula, {"x": x}))
        df = pd.DataFrame(
            spline_dm, columns=["knot{}".format(idx + 1) for idx in range(n_knots)]
        )
    return DesignMatrix(df, name=name)
