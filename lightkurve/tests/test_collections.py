import pytest
from astropy.utils.data import get_pkg_data_filename
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from ..lightcurve import LightCurve
from ..targetpixelfile import KeplerTargetPixelFile
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
    assert(lcc[0] == lc)
    assert(lcc[1] == lc2)
    with pytest.raises(IndexError):
        lcc[50]


def test_collection_setitem():
    """Tests Collection. __setitem__"""
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5),
                    flux_err=np.arange(1, 5), targetid=50000)
    lc2 = LightCurve(time=np.arange(10, 15), flux=np.arange(10, 15),
                     flux_err=np.arange(10, 15), targetid=120334)
    lcc = LightCurveCollection([lc])
    lcc.append(lc2)
    lc3 = LightCurve([1], targetid=55)
    lcc[1] = lc3
    assert(lcc[1] == lc3)
    lcc.append(lc2)
    assert(lcc[2] == lc2)
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
