import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from .. import DesignMatrix, DesignMatrixCollection


def test_designmatrix_basics():
    size, name = 10, 'testmatrix'
    df = pd.DataFrame({'vector1': np.ones(size),
                       'vector2': np.zeros(size),
                       'vector3': np.ones(size)})
    dm = DesignMatrix(df, name=name)
    assert dm.columns[0] == 'vector1'
    assert dm.name == name
    assert dm.shape == (size, 3)
    assert (dm['vector1'] == df['vector1']).all()


def test_designmatrix_from_numpy():
    size = 10
    dm = DesignMatrix(np.ones((size, 2)))
    assert dm.columns == [0, 1]
    assert dm.name == 'unnamed_matrix'
    assert (dm[0] == np.ones(size)).all()


def test_designmatrix_from_dict():
    size = 10
    dm = DesignMatrix({'centroid_col': np.ones(size),
                       'centroid_row': np.ones(size)},
                       name='motion_systematics')
    assert dm.shape == (10, 2)
    assert (dm['centroid_col'] == np.ones(size)).all()


def test_designmatrixcollection_api():
    size = 5
    dm1 = DesignMatrix(np.ones((size, 1)), columns=['col1'], name='matrix1')
    dm2 = DesignMatrix(np.zeros((size, 2)), columns=['col2', 'col3'], name='matrix2')
    dmc = DesignMatrixCollection([dm1, dm2])
    assert_array_equal(dmc['matrix1'].values, dm1.values)
    assert_array_equal(dmc['matrix2'].values, dm2.values)
    assert_array_equal(dmc.values, np.hstack((dm1.values, dm2.values)))
