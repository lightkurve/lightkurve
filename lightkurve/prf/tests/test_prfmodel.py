"""Test the features of the lightkurve.prf.prfmodels module."""
from __future__ import division, print_function

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ...prf import KeplerPRF, SimpleKeplerPRF
from ...targetpixelfile import KeplerTargetPixelFile


def test_prf_normalization():
    """Does the PRF model integrate to the requested flux across the focal plane?"""
    for channel in [1, 20, 40, 60, 84]:
        for col in [123, 678]:
            for row in [234, 789]:
                shape = (18, 14)
                flux = 100
                prf = KeplerPRF(channel=channel, column=col, row=row, shape=shape)
                prf_sum = prf.evaluate(col + shape[0]/2, row + shape[1]/2, flux, 1, 1, 0).sum()
                assert np.isclose(prf_sum, flux, rtol=0.1)


def test_simple_kepler_prf():
    """Ensures that concentric PRFs have the same values."""
    prf_1 = SimpleKeplerPRF(channel=16, shape=[20, 20], column=0, row=0)
    prf_2 = SimpleKeplerPRF(channel=16, shape=[10, 10], column=5, row=5)
    for c in [10, 8, 10, 7]:
        for r in [10, 10, 7, 7]:
            assert_allclose(prf_2(center_col=c, center_row=r, flux=1),
                            prf_1(center_col=c, center_row=r, flux=1)[5:15, 5:15],
                            rtol=1e-5)


@pytest.mark.remote_data
def test_simple_kepler_prf_interpolation_consistency():
    """Ensures that the interpolated prf is consistent with calibration files.
    """
    sprf = SimpleKeplerPRF(channel=56, shape=[15, 15], column=0, row=0)
    cal_prf = fits.open("http://archive.stsci.edu/missions/kepler/fpc/prf/"
                        "extracted/kplr16.4_2011265_prf.fits")
    cal_prf_subsampled = cal_prf[-1].data[25::50, 25::50]
    cal_prf_subsampled_normalized = cal_prf_subsampled / (cal_prf[-1].data.sum() * 0.02 ** 2)
    sprf_data = sprf(center_col=7.5, center_row=7.5, flux=1)
    np.isclose(np.sum(np.abs(sprf_data - cal_prf_subsampled_normalized)), 0)


def test_get_model_prf():
    tpf_fn = get_pkg_data_filename("../../tests/data/test-tpf-star.fits")
    tpf = KeplerTargetPixelFile(tpf_fn)

    prf = KeplerPRF(channel=tpf.channel, shape=tpf.shape[1:],
                    column=tpf.column, row=tpf.row)
    prf_from_tpf = tpf.get_prf_model()

    assert type(prf) == type(prf_from_tpf)
    assert prf.channel == prf_from_tpf.channel
    assert prf.shape == prf_from_tpf.shape
    assert prf.column == prf_from_tpf.column
    assert prf.row == prf_from_tpf.row
