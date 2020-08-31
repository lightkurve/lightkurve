import pytest
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

<<<<<<< HEAD
from ..lightcurve import LightCurve
from ..search import search_lightcurve
from ..targetpixelfile import KeplerTargetPixelFile
=======
from ..lightcurve import LightCurve, TessLightCurve
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
>>>>>>> 1608e57... Collection.sector - test; handle boundary cases
from ..collections import LightCurveCollection, TargetPixelFileCollection

filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")


def test_collection_init():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
    lc2 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15), flux_err=np.arange(10, 15))
    lcc = LightCurveCollection([lc, lc2])
    assert(len(lcc) == 2)
    assert(lcc.data == [lc, lc2])
    str(lcc)  # Does repr work?
    lcc.plot()
    plt.close('all')


def test_collection_append():
    """Does Collection.append() work?"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=500)
    lc2 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=100)
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    assert(len(lcc) == 2)

def test_collection_stitch():
    """Does Collection.stitch() work?"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.ones(4))
    lc2 = LightCurve(time=np.arange(5, 16), flux=np.ones(11))
    lcc = LightCurveCollection([lc, lc2])
    lc_stitched = lcc.stitch()
    assert(len(lc_stitched.flux) == 15)
    lc_stitched2 = lcc.stitch(corrector_func=lambda x: x*2)
    assert_array_equal(lc_stitched.flux*2, lc_stitched2.flux)

def test_collection_getitem():
    """Tests Collection.__getitem__"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=50000)
    lc2 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=120334)
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    assert((lcc[0] == lc).all())
    assert((lcc[1] == lc2).all())
    with pytest.raises(IndexError):
        lcc[50]

def test_collection_getitem_by_boolean_array():
    """Tests Collection.__getitem__ , case the argument is a mask, i.e, indexed by boolean array"""
    lc0 = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=50000)
    lc1 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=120334)
    lc2 = LightCurve(time=np.arange(15, 20), flux=np.arange(15, 20),
                     flux_err=np.arange(15, 20), targetid=23456)
    lcc = LightCurveCollection([lc0, lc1, lc2])

    lcc_f = lcc[[True, False, True]]
    assert(lcc_f.data == [lc0, lc2])
    assert(type(lcc_f), LightCurveCollection)

    # boundary case: 1 element
    lcc_f = lcc[[False, True, False]]
    assert(lcc_f.data == [lc1])
    # boundary case: no element
    lcc_f = lcc[[False, False, False]]
    assert(lcc_f.data == [])
    # other array-like input: tuple
    lcc_f = lcc[(True, False, True)]
    assert(lcc_f.data == [lc0, lc2])
    # other array-like input: ndarray
    lcc_f = lcc[np.array([True, False, True])]
    assert(lcc_f.data == [lc0, lc2])

    # boundary case: mask length not matching - shorter
    with pytest.raises(IndexError):
        lcc[[True, False]]

    # boundary case: mask length not matching - longer
    with pytest.raises(IndexError):
        lcc[[True, False, True, True]]

def test_collection_getitem_by_other_array():
    """Tests Collection.__getitem__ , case the argument an non-boolean array"""
    lc0 = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=50000)
    lc1 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=120334)
    lc2 = LightCurve(time=np.arange(15, 20), flux=np.arange(15, 20),
                     flux_err=np.arange(15, 20), targetid=23456)
    lcc = LightCurveCollection([lc0, lc1, lc2])

    # case: an int array-like, follow ndarray behavior
    lcc_f = lcc[[2, 0]]
    assert(lcc_f.data == [lc2, lc0])
    lcc_f = lcc[np.array([2, 0])]
    assert(lcc_f.data == [lc2, lc0])
    # support other int types in np too
    lcc_f = lcc[np.array([np.int64(2), np.uint8(0)])]
    assert(lcc_f.data == [lc2, lc0])
    # boundary condition: True / False is interpreted as 1/0 in an bool/int mixed array-like
    lcc_f = lcc[[True, False, 2]]
    assert(lcc_f.data == [lc1, lc0, lc2])
    # boundary condition: some index is out of bound
    with pytest.raises(IndexError):
        lcc[[2, 99]]
    # boundary conditions: array-like of neither bool or int, follow ndarray behavior
    with pytest.raises(IndexError):
        lcc[['abc', 'def']]
    with pytest.raises(IndexError):
        lcc[[True, 'def']]

def test_collection_setitem():
    """Tests Collection. __setitem__"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=50000)
    lc2 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=120334)
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    lc3 = LightCurve(time=[1], targetid=55)
    lcc[1] = lc3
    assert(lcc[1].time == lc3.time)
    lcc.append(lc2)
    assert((lcc[2].time == lc2.time).all())
    with pytest.raises(IndexError):
        lcc[51] = 10


def test_tpfcollection():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center)
    tpfc = TargetPixelFileCollection([tpf, tpf2])
    assert(len(tpfc) == 2)
    assert(tpfc.data == [tpf, tpf2])
    tpfc.append(tpf2)
    assert(len(tpfc) == 3)
    assert(tpfc[0] == tpf)
    assert(tpfc[1] == tpf2)
    assert(tpfc[2] == tpf2)
    with pytest.raises(IndexError):
        tpfc[51]
    # ensure index by boolean array also works for TPFs
    tpfc_f = tpfc[[False, True, True]]
    assert(tpfc_f.data == [tpf2, tpf2])
    assert(type(tpfc_f), TargetPixelFileCollection)
    # Test __setitem__
    tpf3 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=55)
    tpfc[1] = tpf3
    assert(tpfc[1] == tpf3)
    tpfc.append(tpf2)
    assert(tpfc[2] == tpf2)
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
    plt.close('all')


@pytest.mark.remote_data
def test_stitch_repr():
    """Regression test for #884."""
    lc = search_lightcurve("Pi Men", mission='TESS', sector=1).download()
    # The line below used to raise `ValueError: Unable to parse format string
    # "{:10d}" for entry "70445.0" in column "cadenceno"`
    LightCurveCollection((lc,lc)).stitch().__repr__()


def test_accessor_tess_sector():
    lc0 = TessLightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                         flux_err=np.arange(1, 5), targetid=50000)
    lc0.sector = 14
    lc1 = TessLightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                         flux_err=np.arange(10, 15), targetid=120334)
    lc1.sector = 26
    lcc = LightCurveCollection([lc0, lc1])
    assert(lcc.sector == [14, 26])

    # boundary condition: some lightcurve objects do not have sector
    lc2 = LightCurve(time=np.arange(15, 20), flux=np.arange(15, 20),
                     flux_err=np.arange(15, 20), targetid=23456)
    lcc.append(lc2)
    assert(lcc.sector == [14, 26, None])

    # ensure it works for TPFs too.
    tpf = TessTargetPixelFile(filename_tpf_all_zeros)
    tpf.hdu[0].header['SECTOR'] = 23
    tpf2 = TessTargetPixelFile(filename_tpf_one_center)
    # tpf2 has no sector defined
    tpf3 = TessTargetPixelFile(filename_tpf_one_center)
    tpf3.hdu[0].header['SECTOR'] = 1
    tpfc = TargetPixelFileCollection([tpf, tpf2, tpf3])
    assert(tpfc.sector == [23, None, 1])

