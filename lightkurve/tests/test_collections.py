import pytest
from ..lightcurve import LightCurve
from ..targetpixelfile import KeplerTargetPixelFile
from ..collection import *
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import logging

log = logging.getLogger(__name__)
filename_tpf_all_zeros = get_pkg_data_filename("data/test-tpf-all-zeros.fits")
filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")
def test_collection_init():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5))
    lc2 = LightCurve(time=np.arange(10,15), flux=np.arange(10,15), flux_err=np.arange(10, 15))

    lcc = LightCurveCollection([lc, lc2])

    assert(len(lcc) == 2)
    assert(lcc.data == [lc, lc2])
    assert(type(lcc.k_id) == dict)


def test_collection_append():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5), targetid=500)
    lc2 = LightCurve(time=np.arange(10,15), flux=np.arange(10,15), flux_err=np.arange(10, 15), targetid=100)

    lcc = LightCurveCollection([lc])
    lcc.append(lc2)

    assert(len(lcc) == 2)

    #What happens when I try to append the same obj twice?
    #An error should probably occur
    with pytest.raises(AttributeError):
        lcc.append(lc2)

def test_collection_getitem():
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5), targetid=50000)
    lc2 = LightCurve(time=np.arange(10,15), flux=np.arange(10,15), flux_err=np.arange(10, 15), targetid=120334)

    lcc = LightCurveCollection([lc])
    lcc.append(lc2)

    assert(lcc[0] == lc)
    assert(lcc[1] == lc2)

    assert(lcc[lc.targetid] == lc)
    assert(lcc[lc2.targetid] == lc2)

    with pytest.raises(KeyError):
        #Tests __getitem__
        lcc[51] = 10

def test_collection_assignment():
    #Tests for __getitem__, __setitem__
    lc = LightCurve(time=np.arange(1, 5), flux=np.arange(1, 5), flux_err=np.arange(1, 5), targetid=50000)
    lc2 = LightCurve(time=np.arange(10,15), flux=np.arange(10,15), flux_err=np.arange(10, 15), targetid=120334)

    lcc = LightCurveCollection([lc])
    lcc.append(lc2)

    lc3 = LightCurve([1], targetid=55)
    lcc[1] = lc3

    assert(lcc[1] == lc3)

    lcc.append(lc2)
    assert(lcc[2] == lc2)

def test_tpfcollection_init():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center)

    tpfc = TargetPixelFileCollection([tpf, tpf2])

    assert(len(tpfc) == 2)
    assert(tpfc.data == [tpf, tpf2])
    assert(type(tpfc.k_id) == dict)


def test_tpfcollection_append():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, targetid=500)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=100)
    tpfc = TargetPixelFileCollection([tpf])
    tpfc.append(tpf2)

    assert(len(tpfc) == 2)

    #What happens when I try to append the same obj twice?
    #An error should probably occur
    with pytest.raises(AttributeError):
        tpfc.append(tpf2)

def test_tpfcollection_getitem():
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, targetid=50000)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=120334)

    tpfc = TargetPixelFileCollection([tpf])
    tpfc.append(tpf2)

    assert(tpfc[0] == tpf)
    assert(tpfc[1] == tpf2)

    assert(tpfc[tpf.targetid] == tpf)
    assert(tpfc[tpf2.targetid] == tpf2)

    with pytest.raises(KeyError):
        #Tests __getitem__
        tpfc[51] = 10

def test_tpfcollection_assignment():
    #Tests for __getitem__, __setitem__
    tpf = KeplerTargetPixelFile(filename_tpf_all_zeros, targetid=50000)
    tpf2 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=120334)

    tpfc = TargetPixelFileCollection([tpf])
    tpfc.append(tpf2)

    tpf3 = KeplerTargetPixelFile(filename_tpf_one_center, targetid=55)
    
    tpfc[1] = tpf3

    assert(tpfc[1] == tpf3)

    tpfc.append(tpf2)
    assert(tpfc[2] == tpf2)
