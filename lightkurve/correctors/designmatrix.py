import numpy as np
import pandas as pd


__all__ = ['DesignMatrix', 'DesignMatrixCollection']


class DesignMatrix():
    """A matrix with named columns for use by systematics removal methods."""
    def __init__(self, df, columns=None, name='unnamed_matrix'):
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if columns is not None:
            df.columns = columns
        self.df = df
        self.name = name

    @property
    def columns(self):
        return self.df.columns.to_list()

    @property
    def shape(self):
        return self.df.shape

    @property
    def values(self):
        return self.df.values

    def __getitem__(self, key):
        return self.df[key]


class DesignMatrixCollection():
    """A set of design matrixes."""
    def __init__(self, matrixes):
        self.matrixes = matrixes

    @property
    def values(self):
        return np.hstack((m.values for m in self.matrixes))

    def __getitem__(self, key):
        try:
            return self.matrixes[key]
        except Exception:
            arg = np.argwhere([m.name == key for m in self.matrixes])
            return self.matrixes[arg[0][0]]
