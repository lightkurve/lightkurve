import warnings

import pytest
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.masked import Masked
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from lightkurve.lightcurve import LightCurve, KeplerLightCurve, TessLightCurve
from lightkurve.search import search_lightcurve
from lightkurve.targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from lightkurve.collections import LightCurveCollection, TargetPixelFileCollection
from lightkurve.utils import LightkurveWarning

filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")


def test_collection_init():
    lc = LightCurve(
        time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5)
    )
    lc2 = LightCurve(
        time=np.arange(10, 15), flux=np.arange(10, 15), flux_err=np.arange(10, 15)
    )
    lcc = LightCurveCollection([lc, lc2])
    assert len(lcc) == 2
    assert lcc.data == [lc, lc2]
    str(lcc)  # Does repr work?
    lcc.plot()
    plt.close("all")


def test_collection_append():
    """Does Collection.append() work?"""
    lc = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=500,
    )
    lc2 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=100,
    )
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    assert len(lcc) == 2


def test_collection_stitch():
    """Does Collection.stitch() work?"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.ones(4))
    lc2 = LightCurve(time=np.arange(5, 16), flux=np.ones(11))
    lcc = LightCurveCollection([lc, lc2])
    lc_stitched = lcc.stitch()
    assert len(lc_stitched.flux) == 15
    lc_stitched2 = lcc.stitch(corrector_func=lambda x: x * 2)
    assert_array_equal(lc_stitched.flux * 2, lc_stitched2.flux)


def test_collection_stitch_with_masked_values():
    """Test https://github.com/lightkurve/lightkurve/issues/1178 """
    lc = LightCurve(time=np.arange(1, 5), flux=np.ones(4))
    lc2 = LightCurve(
        time=np.arange(5, 9),
        flux=Masked([11, 11, np.nan, 11], mask=[False, False, True, False]),
    )
    lc_stitched = LightCurveCollection([lc, lc2]).stitch()
    assert len(lc_stitched.flux) == 8

    # ensure order (whether the first lc is masked or not) does not matter
    lc3 = LightCurve(time=np.arange(9, 13), flux=np.ones(4))
    lc_stitched = LightCurveCollection([lc2, lc3]).stitch()
    assert len(lc_stitched.flux) == 8


def test_collection_getitem():
    """Tests Collection.__getitem__"""
    lc = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc2 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    assert (lcc[0] == lc).all()
    assert (lcc[1] == lc2).all()
    with pytest.raises(IndexError):
        lcc[50]


def test_collection_getitem_by_boolean_array():
    """Tests Collection.__getitem__ , case the argument is a mask, i.e, indexed by boolean array"""
    lc0 = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc1 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc2 = LightCurve(
        time=np.arange(15, 20),
        flux=np.arange(15, 20),
        flux_err=np.arange(15, 20),
        targetid=23456,
    )
    lcc = LightCurveCollection([lc0, lc1, lc2])

    lcc_f = lcc[[True, False, True]]
    assert lcc_f.data == [lc0, lc2]
    assert type(lcc_f) is LightCurveCollection

    # boundary case: 1 element
    lcc_f = lcc[[False, True, False]]
    assert lcc_f.data == [lc1]
    # boundary case: no element
    lcc_f = lcc[[False, False, False]]
    assert lcc_f.data == []
    # other array-like input: tuple
    lcc_f = lcc[(True, False, True)]
    assert lcc_f.data == [lc0, lc2]
    # other array-like input: ndarray
    lcc_f = lcc[np.array([True, False, True])]
    assert lcc_f.data == [lc0, lc2]

    # boundary case: mask length not matching - shorter
    with pytest.raises(IndexError):
        lcc[[True, False]]

    # boundary case: mask length not matching - longer
    with pytest.raises(IndexError):
        lcc[[True, False, True, True]]


def test_collection_getitem_by_other_array():
    """Tests Collection.__getitem__ , case the argument an non-boolean array"""
    lc0 = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc1 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc2 = LightCurve(
        time=np.arange(15, 20),
        flux=np.arange(15, 20),
        flux_err=np.arange(15, 20),
        targetid=23456,
    )
    lcc = LightCurveCollection([lc0, lc1, lc2])

    # case: an int array-like, follow ndarray behavior
    lcc_f = lcc[[2, 0]]
    assert lcc_f.data == [lc2, lc0]
    lcc_f = lcc[np.array([2, 0])]
    assert lcc_f.data == [lc2, lc0]
    # support other int types in np too
    lcc_f = lcc[np.array([np.int64(2), np.uint8(0)])]
    assert lcc_f.data == [lc2, lc0]
    # boundary condition: True / False is interpreted as 1/0 in an bool/int mixed array-like
    lcc_f = lcc[[True, False, 2]]
    assert lcc_f.data == [lc1, lc0, lc2]
    # boundary condition: some index is out of bound
    with pytest.raises(IndexError):
        lcc[[2, 99]]
    # boundary conditions: array-like of neither bool or int, follow ndarray behavior
    with pytest.raises(IndexError):
        lcc[["abc", "def"]]
    with pytest.raises(IndexError):
        lcc[[True, "def"]]


def test_collection_getitem_by_slices():
    """Tests Collection.__getitem__ by slices"""
    lc0 = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc1 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc2 = LightCurve(
        time=np.arange(15, 20),
        flux=np.arange(15, 20),
        flux_err=np.arange(15, 20),
        targetid=23456,
    )
    lcc = LightCurveCollection([lc0, lc1, lc2])

    assert lcc[:2].data == [lc0, lc1]

    assert lcc[1:999].data == [lc1, lc2]


def test_collection_setitem():
    """Tests Collection. __setitem__"""
    lc = LightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc2 = LightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    lc3 = LightCurve(time=[1], targetid=55)
    lcc[1] = lc3
    assert lcc[1].time == lc3.time
    lcc.append(lc2)
    assert (lcc[2].time == lc2.time).all()
    with pytest.raises(IndexError):
        lcc[51] = 10


def test_tpfcollection():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center)
    tpfc = TargetPixelFileCollection([tpf, tpf2])
    assert len(tpfc) == 2
    assert tpfc.data == [tpf, tpf2]
    tpfc.append(tpf2)
    assert len(tpfc) == 3
    assert tpfc[0] == tpf
    assert tpfc[1] == tpf2
    assert tpfc[2] == tpf2
    with pytest.raises(IndexError):
        tpfc[51]
    # ensure index by boolean array also works for TPFs
    tpfc_f = tpfc[[False, True, True]]
    assert tpfc_f.data == [tpf2, tpf2]
    assert type(tpfc_f) is TargetPixelFileCollection
    # Test __setitem__
    tpf3 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=55)
    tpfc[1] = tpf3
    assert tpfc[1] == tpf3
    tpfc.append(tpf2)
    assert tpfc[2] == tpf2
    str(tpfc)  # Regression test for #564


def test_tpfcollection_plot():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center)
    # Does plotting work with 3 TPFs?
    coll = TargetPixelFileCollection([tpf, tpf2, tpf2])
    coll.plot()
    # Does plotting work with one TPF?
    coll = TargetPixelFileCollection([tpf])
    coll.plot()
    plt.close("all")


@pytest.mark.remote_data
def test_stitch_repr():
    """Regression test for #884."""
    lc = search_lightcurve("Pi Men", mission="TESS", author="SPOC", sector=1).download()
    # The line below used to raise `ValueError: Unable to parse format string
    # "{:10d}" for entry "70445.0" in column "cadenceno"`
    LightCurveCollection((lc, lc)).stitch().__repr__()


def test_accessor_tess_sector():
    lc0 = TessLightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc0.meta["SECTOR"] = 14
    lc1 = TessLightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc1.meta["SECTOR"] = 26
    lcc = LightCurveCollection([lc0, lc1])
    assert (lcc.sector == [14, 26]).all()
    # The sector accessor can be used to generate boolean array
    # to support filter collection by sector
    assert ((lcc.sector == 26) == [False, True]).all()
    assert ((lcc.sector < 20) == [True, False]).all()

    # boundary condition: some lightcurve objects do not have sector
    lc2 = LightCurve(
        time=np.arange(15, 20),
        flux=np.arange(15, 20),
        flux_err=np.arange(15, 20),
        targetid=23456,
    )
    lcc.append(lc2)
    # expecting [14, 26, np.nan], need 2 asserts to do it.
    assert (lcc.sector[:-1] == [14, 26]).all()
    assert np.isnan(lcc.sector[-1])
    # The sector accessor can be used to generate boolean array
    # to support filter collection by sector
    assert ((lcc.sector == 26) == [False, True, False]).all()
    assert ((lcc.sector < 20) == [True, False, False]).all()

    # ensure it works for TPFs too.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LightkurveWarning)
        # Ignore "A Kepler data product is being opened using the `TessTargetPixelFile` class"
        # the test only cares about the SECTOR header that it sets.
        tpf = TessTargetPixelFile(filename_tpf_all_zeros)
        tpf.hdu[0].header["SECTOR"] = 23
        tpf2 = TessTargetPixelFile(filename_tpf_one_center)
        # tpf2 has no sector defined
        tpf3 = TessTargetPixelFile(filename_tpf_one_center)
        tpf3.hdu[0].header["SECTOR"] = 1
    tpfc = TargetPixelFileCollection([tpf, tpf2, tpf3])
    assert (tpfc.sector == [23, None, 1]).all()


def test_accessor_kepler_quarter():
    # scaled down version of tess sector test, as they share the same codepath
    lc0 = KeplerLightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc0.meta["QUARTER"] = 2
    lc1 = KeplerLightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc1.meta["QUARTER"] = 1
    lcc = LightCurveCollection([lc0, lc1])
    assert (lcc.quarter == [2, 1]).all()

    # ensure it works for TPFs too.
    tpf0 = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf0.hdu[0].header["QUARTER"] = 2
    tpf1 = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf1.hdu[0].header["QUARTER"] = 1
    tpfc = TargetPixelFileCollection([tpf0, tpf1])
    assert (tpfc.quarter == [2, 1]).all()


def test_accessor_k2_campaign():
    # scaled down version of tess sector test, as they share the same codepath
    lc0 = KeplerLightCurve(
        time=np.arange(1, 5),
        flux=np.arange(1, 5),
        flux_err=np.arange(1, 5),
        targetid=50000,
    )
    lc0.meta["CAMPAIGN"] = 2
    lc1 = KeplerLightCurve(
        time=np.arange(10, 15),
        flux=np.arange(10, 15),
        flux_err=np.arange(10, 15),
        targetid=120334,
    )
    lc1.meta["CAMPAIGN"] = 1
    lcc = LightCurveCollection([lc0, lc1])
    assert (lcc.campaign == [2, 1]).all()

    # ensure it works for TPFs too.
    tpf0 = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf0.hdu[0].header["CAMPAIGN"] = 2
    tpf1 = KeplerTargetPixelFile(filename_tpf_one_center)
    tpf1.hdu[0].header["CAMPAIGN"] = 1
    tpfc = TargetPixelFileCollection([tpf0, tpf1])
    assert (tpfc.campaign == [2, 1]).all()


def test_unmergeable_columns():
    """Regression test for #954 and #1015."""
    lc1 = LightCurve(data={'time': [1,2,3], 'x': [1,2,3]})
    lc2 = LightCurve(data={'time': [1,2,3], 'x': [1,2,3]*u.electron/u.second})
    with pytest.warns(LightkurveWarning, match="column types are incompatible"):
        LightCurveCollection([lc1, lc2]).stitch()
    with pytest.warns(LightkurveWarning, match="column types are incompatible"):
        lc1.append(lc2)
