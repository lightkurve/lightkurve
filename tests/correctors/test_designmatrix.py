import pytest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from scipy.sparse import csr_matrix

from lightkurve.correctors import (
    DesignMatrix,
    DesignMatrixCollection,
    SparseDesignMatrix,
)
from lightkurve import LightkurveWarning


def test_designmatrix_basics():
    """Can we create a design matrix from a dataframe?"""
    size, name = 10, "testmatrix"
    df = pd.DataFrame(
        {"vector1": np.ones(size), "vector2": np.zeros(size), "vector3": np.ones(size)}
    )
    dm = DesignMatrix(df, name=name)
    assert dm.columns == ["vector1", "vector2", "vector3"]
    assert dm.name == name
    assert dm.shape == (size, 3)
    assert (dm["vector1"] == df["vector1"]).all()
    dm.plot()
    dm.plot_priors()
    assert dm.append_constant().shape == (size, 4)  # expect one column more
    assert dm.pca(nterms=2).shape == (size, 2)  # expect one column less
    # vector2 is all zeros, and the split index is the last row, so only the
    # two nonzero pieces of the first segment remain.
    assert dm.split([10]).shape == (size, 2)
    dm.__repr__()

    dm = DesignMatrix(df, name=name)
    dm.append_constant(inplace=True)
    assert dm.shape == (size, 4)  # expect one column more

    dm = DesignMatrix(df, name=name)
    dm.split([10], inplace=True)
    assert dm.shape == (size, 2)


def test_designmatrix_from_numpy():
    """Can we create a design matrix from an ndarray?"""
    size = 10
    dm = DesignMatrix(np.ones((size, 2)))
    assert dm.columns == [0, 1]
    assert dm.name == "unnamed_matrix"
    assert (dm[0] == np.ones(size)).all()


def test_designmatrix_from_dict():
    """Can we create a design matrix from a dictionary?"""
    size = 10
    dm = DesignMatrix(
        {"centroid_col": np.ones(size), "centroid_row": np.ones(size)},
        name="motion_systematics",
    )
    assert dm.shape == (size, 2)
    assert (dm["centroid_col"] == np.ones(size)).all()


def test_split():
    """Can we split a design matrix correctly?"""
    dm = DesignMatrix({"a": np.linspace(0, 9, 10), "b": np.linspace(100, 109, 10)})
    # Do we retrieve the correct shape?
    assert dm.shape == (10, 2)
    assert dm.split(2).shape == (10, 4)
    assert dm.split([2, 8]).shape == (10, 6)
    # Are the new areas padded with zeros?
    assert (dm.split([2, 8]).values[2:, 0:2] == 0).all()
    assert (dm.split([2, 8]).values[:8, 4:] == 0).all()
    # Are all the column names unique?
    assert len(set(dm.split(2).columns)) == 4


def test_split_drops_all_zero_columns():
    """An all-zero piece is omitted, as SparseDesignMatrix.split already does."""
    values = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    prior_mu = [0.2, 0.4]
    prior_sigma = [1.0, 2.0]
    dm = DesignMatrix(
        values, columns=["a", "b"], prior_mu=prior_mu, prior_sigma=prior_sigma
    )
    split = dm.split([3])
    assert split.shape == (6, 3)
    assert list(split.columns) == ["a 1", "b 1", "a 2"]
    assert not (split.values == 0).all(axis=0).any()
    assert_array_equal(split.prior_mu, [0.2, 0.4, 0.2])
    assert_array_equal(split.prior_sigma, [1.0, 2.0, 1.0])
    assert dm.shape == (6, 2)

    sparse = SparseDesignMatrix(
        csr_matrix(values),
        columns=["a", "b"],
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
    ).split([3])
    assert_array_equal(split.values, np.asarray(sparse.values))
    assert_array_equal(split.prior_mu, sparse.prior_mu)
    assert_array_equal(split.prior_sigma, sparse.prior_sigma)

    dm.split([3], inplace=True)
    assert dm.shape == (6, 3)
    assert_array_equal(dm.prior_mu, [0.2, 0.4, 0.2])


def test_standardize():
    """Verifies DesignMatrix.standardize()"""
    # A column with zero standard deviation remains unchanged
    dm = DesignMatrix({"const": np.ones(10)})
    assert (dm.standardize()["const"] == dm["const"]).all()
    # Normally-distributed columns will become Normal(0, 1)
    dm = DesignMatrix({"normal": np.random.normal(loc=5, scale=3, size=100)})
    assert np.round(np.median(dm.standardize()["normal"]), 3) == 0
    assert np.round(np.std(dm.standardize()["normal"]), 1) == 1
    dm.standardize(inplace=True)


def test_pca():
    """Verifies DesignMatrix.pca()"""
    size = 10
    dm = DesignMatrix(
        {
            "a": np.random.normal(10, 20, size),
            "b": np.random.normal(40, 10, size),
            "c": np.random.normal(60, 5, size),
        }
    )
    for nterms in [1, 2, 3]:
        assert dm.pca(nterms=nterms).shape == (size, nterms)


def test_collection_basics():
    """Can we create a design matrix collection?"""
    size = 5
    dm1 = DesignMatrix(np.ones((size, 1)), columns=["col1"], name="matrix1")
    dm2 = DesignMatrix(np.zeros((size, 2)), columns=["col2", "col3"], name="matrix2")

    dmc = DesignMatrixCollection([dm1, dm2])
    assert_array_equal(dmc["matrix1"].values, dm1.values)
    assert_array_equal(dmc["matrix2"].values, dm2.values)
    assert_array_equal(dmc.values, np.hstack((dm1.values, dm2.values)))
    dmc.plot()
    dmc.__repr__()

    dmc = dm1.collect(dm2)
    assert_array_equal(dmc["matrix1"].values, dm1.values)
    assert_array_equal(dmc["matrix2"].values, dm2.values)
    assert_array_equal(dmc.values, np.hstack((dm1.values, dm2.values)))

    assert isinstance(dmc.to_designmatrix(), DesignMatrix)


def test_designmatrix_rank():
    """Does DesignMatrix issue a low-rank warning when justified?"""
    warnings.simplefilter("always")

    # Good rank
    dm = DesignMatrix({"a": [1, 2, 3]})
    assert dm.rank == 1
    dm.validate(rank=True)  # Should not raise a warning

    # Bad rank
    with pytest.warns(LightkurveWarning, match="rank"):
        dm = DesignMatrix(
            {
                "a": [1, 2, 3],
                "b": [1, 1, 1],
                "c": [1, 1, 1],
                "d": [1, 1, 1],
                "e": [3, 4, 5],
            }
        )
        assert dm.rank == 2
        dm.validate(rank=True)  # Should raise a warning
