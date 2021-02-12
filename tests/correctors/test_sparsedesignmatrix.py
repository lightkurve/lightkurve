import pytest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from .. import SparseDesignMatrix, SparseDesignMatrixCollection, DesignMatrix, DesignMatrixCollection
from ... import LightkurveWarning
from ..designmatrix import create_sparse_spline_matrix, create_spline_matrix


from scipy import sparse


def test_designmatrix_basics():
    """Can we create a design matrix from a dataframe?"""
    size, name = 10, 'testmatrix'
    df = pd.DataFrame({'vector1': np.ones(size),
                       'vector2': np.arange(size),
                       'vector3': np.arange(size)**2})
    X = sparse.csr_matrix(np.asarray(df))
    dm = SparseDesignMatrix(X, name=name, columns=['vector1', 'vector2', 'vector3'])
    assert dm.columns == ['vector1', 'vector2', 'vector3']
    assert dm.name == name
    assert dm.shape == (size, 3)
    dm.plot()
    dm.plot_priors()
    assert dm.append_constant().shape == (size, 4)  # expect one column more
    assert dm.pca(nterms=2).shape == (size, 2)      # expect one column less
    assert dm.split([5]).shape == (size, 6)        # expect double columns
    dm.__repr__()

    dm = SparseDesignMatrix(X, name=name, columns=['vector1', 'vector2', 'vector3'])
    dm.append_constant(inplace=True)
    assert dm.shape == (size, 4)  # expect one column more

    dm = SparseDesignMatrix(X, name=name, columns=['vector1', 'vector2', 'vector3'])
    dm.split([5], inplace=True)
    assert dm.shape == (size, 6)        # expect double columns


def test_split():
    """Can we split a design matrix correctly?"""
    X = sparse.csr_matrix(np.vstack([np.linspace(0, 9, 10), np.linspace(100, 109, 10)]).T)
    dm = SparseDesignMatrix(X, columns=['a', 'b'])
    # Do we retrieve the correct shape?
    assert dm.shape == (10, 2)
    assert dm.split(2).shape == (10, 4)
    assert dm.split([2,8]).shape == (10, 6)
    # Are the new areas padded with zeros?
    assert (dm.split([2,8]).values[2:, 0:2] == 0).all()
    assert (dm.split([2,8]).values[:8, 4:] == 0).all()
    # Are all the column names unique?
    assert len(set(dm.split(4).columns)) == 4


def test_standardize():
    """Verifies DesignMatrix.standardize()"""
    # A column with zero standard deviation remains unchanged
    X = sparse.csr_matrix(np.vstack([np.ones(10)]).T)
    dm = SparseDesignMatrix(X, columns=['const'])
    assert (dm.standardize()['const'] == dm['const']).all()
    # Normally-distributed columns will become Normal(0, 1)
    X = sparse.csr_matrix(np.vstack([ np.random.normal(loc=5, scale=3, size=100)]).T)
    dm = SparseDesignMatrix(X, columns=['normal'])

    assert np.round(np.mean(dm.standardize()['normal']), 3) == 0
    assert np.round(np.std(dm.standardize()['normal']), 1) == 1
    dm.standardize(inplace=True)

def test_pca():
    """Verifies DesignMatrix.pca()"""
    size = 10
    dm = DesignMatrix({'a':np.random.normal(10, 20, size),
                       'b':np.random.normal(40, 10, size),
                       'c':np.random.normal(60, 5, size)}).to_sparse()
    for nterms in [1, 2, 3]:
        assert dm.pca(nterms=nterms).shape == (size, nterms)


def test_collection_basics():
    """Can we create a design matrix collection?"""
    size = 5
    dm1 = DesignMatrix(np.ones((size, 1)), columns=['col1'], name='matrix1').to_sparse()
    dm2 = DesignMatrix(np.zeros((size, 2)), columns=['col2', 'col3'], name='matrix2').to_sparse()

    dmc = SparseDesignMatrixCollection([dm1, dm2])
    assert_array_equal(dmc['matrix1'].values, dm1.values)
    assert_array_equal(dmc['matrix2'].values, dm2.values)
    assert_array_equal(dmc.values, np.hstack((dm1.values, dm2.values)))
    dmc.plot()
    dmc.__repr__()

    dmc = dm1.collect(dm2)
    assert_array_equal(dmc['matrix1'].values, dm1.values)
    assert_array_equal(dmc['matrix2'].values, dm2.values)
    assert_array_equal(dmc.values, np.hstack((dm1.values, dm2.values)))

    """Can we create a design matrix collection when one is sparse?"""
    size = 5
    dm1 = DesignMatrix(np.ones((size, 1)), columns=['col1'], name='matrix1')
    dm2 = DesignMatrix(np.zeros((size, 2)), columns=['col2', 'col3'], name='matrix2').to_sparse()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(LightkurveWarning, match='Sparse matrices will be converted to dense matrices.'):
            dmc = DesignMatrixCollection([dm1, dm2])
            assert not np.any([sparse.issparse(d.X) for d in dmc])

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(LightkurveWarning, match='Dense matrices will be converted to sparse matrices.'):
            dmc = SparseDesignMatrixCollection([dm1, dm2])
            assert np.all([sparse.issparse(d.X) for d in dmc])

    dmc.plot()
    dmc.__repr__()

    assert isinstance(dmc.to_designmatrix(), SparseDesignMatrix)


def test_designmatrix_rank():
    """Does DesignMatrix issue a low-rank warning when justified?"""
    warnings.simplefilter("always")

    # Good rank
    dm = DesignMatrix({'a': [1, 2, 3]}).to_sparse()
    assert dm.rank == 1
    dm.validate(rank=True)  # Should not raise a warning

    # Bad rank
    with pytest.warns(LightkurveWarning, match='rank'):
        dm = DesignMatrix({'a': [1, 2, 3], 'b': [1, 1, 1], 'c': [1, 1, 1],
                           'd': [1, 1, 1], 'e': [3, 4, 5]})
        dm.validate(rank=True) # Should raise a warning
    dm = dm.to_sparse()
    assert dm.rank == 2
    with pytest.warns(LightkurveWarning, match='rank'):
        dm.validate(rank=True)


def test_splines():
    """Do splines work as expected?"""
    # Dense and sparse splines should produce the same answer.
    x = np.linspace(0, 1, 100)
    spline_dense = create_spline_matrix(x, knots=[0.1, 0.3, 0.6, 0.9], degree=2)
    spline_sparse = create_sparse_spline_matrix(x, knots=[0.1, 0.3, 0.6, 0.9], degree=2)
    assert np.allclose(spline_dense.values, spline_sparse.values)
    assert isinstance(spline_dense, DesignMatrix)
    assert isinstance(spline_sparse, SparseDesignMatrix)
