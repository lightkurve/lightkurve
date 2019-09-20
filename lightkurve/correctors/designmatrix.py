import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import MPLSTYLE


__all__ = ['DesignMatrix', 'DesignMatrixCollection']


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
    def __init__(self, df, columns=None, name='unnamed_matrix'):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if columns is not None:
            df.columns = columns
        self.df = df
        self.name = name

    def plot(self, ax=None, show_colorbar=True, **kwargs):
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            im = ax.imshow(self.values, **kwargs, origin='bottom')
            ax.set_aspect(self.shape[1]/(1.6*self.shape[0]))
            ax.set_xlabel('Component')
            ax.set_ylabel('X')
            ax.set_title(self.name)
            ax.set_xticks(np.arange(self.shape[1]))
            ax.set_xticklabels([r'${}$'.format(i) for i in self.columns], rotation=40)
            if show_colorbar:
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Component Value')
        return ax


    def split(self, row_indices):
        """Returns a new matrix with the regressors split over multiple columns.

        This method will return a new design matrix containing
        n_columns*len(row_indices) regressors.  This is useful in situations
        where the linear regression can be improved by fitting separate
        coefficients for different contiguous parts of a regressor.

        Parameters
        ----------
        row_indices : iterable of integers
            Every regressor (i.e. column) in the design matrix will be split
            up over multiple columns separated by the indices provided.

        Returns
        -------
        `~lightkurve.correctors.DesignMatrix`
            A new design matrix with shape (n_rows, len(row_indices)*n_columns).
        """
        if isinstance(row_indices, int):
            row_indices = [row_indices]
        # Where do the submatrices begin and end?
        lower_idx = np.append(0, row_indices)
        upper_idx = np.append(row_indices, len(self.df))
        dfs = []
        for idx, a, b in zip(range(len(lower_idx)), lower_idx, upper_idx):
            new_columns = d = dict((val, val + ' {}'.format(idx + 1))
                                    for val in list(self.df.columns))
            dfs.append(self.df[a:b].rename(columns=new_columns))
        new_df = pd.concat(dfs, axis=1).fillna(0)
        return DesignMatrix(new_df)

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
        return np.hstack((m.values for m in self.matrices))

    def plot(self, ax=None):
        temp_dm = DesignMatrix(pd.concat([d.df for d in self], axis=1))
        ax = temp_dm.plot()
        ax.set_title("Design Matrix Collection")
        return

    def split(self, row_indices):
        return DesignMatrixCollection([d.split(row_indices) for d in self])

    def __getitem__(self, key):
        try:
            return self.matrices[key]
        except Exception:
            arg = np.argwhere([m.name == key for m in self.matrices])
            return self.matrices[arg[0][0]]

    def __repr__(self):
        return 'DesginMatrixCollection:\n' + ''.join(['\t{}\n'.format(i.__repr__()) for i in self])
