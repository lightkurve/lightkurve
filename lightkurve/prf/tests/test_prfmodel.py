"""Test the features of the lightkurve.prf.prfmodels module."""
from __future__ import division, print_function
from collections import OrderedDict

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_allclose
import pytest

from ...prf import KeplerPRF, TessPRF, GaussianPRF
from ...targetpixelfile import KeplerTargetPixelFile


def test_kepler_prf_normalization():
    """Does the PRF model integrate to the requested flux across the focal plane?"""
    for channel in [1, 20, 40, 60, 84]:
        for col in [123, 678]:
            for row in [234, 789]:
                shape = (18, 14)
                flux = 100
                prf = KeplerPRF(channel=channel, column=col, row=row, shape=shape)
                prf_sum = prf.evaluate(col + shape[0]/2, row + shape[1]/2, flux, 1, 1, 0).sum()
                assert np.isclose(prf_sum, flux, rtol=0.1)


def test_tess_prf_normalization():
    """Does the PRF model integrate to the requested flux across the focal plane?"""
    for camera in [1, 2, 3, 4]:
        for ccd in [1, 2, 3, 4]:
            shape = (11, 11)
            flux = 100
            prf = TessPRF(camera=camera, ccd=ccd, column=0, row=0, shape=shape)
            prf_sum = prf.evaluate(shape[0]/2, shape[1]/2, flux, 1, 1, 0).sum()
            assert np.isclose(prf_sum, flux, rtol=0.1)


def test_kepler_prf():
    """Ensures that concentric PRFs have the same values."""
    prf_1 = KeplerPRF(channel=16, shape=[20, 20], column=0, row=0)
    prf_2 = KeplerPRF(channel=16, shape=[10, 10], column=5, row=5)
    for c in [10, 8, 10, 7]:
        for r in [10, 10, 7, 7]:
            assert_allclose(prf_2(c, r, 1, 1, 1, 0),
                            prf_1(c, r, 1, 1, 1, 0)[5:15, 5:15],
                            rtol=1e-5)


def test_tess_prf():
    """Ensures that concentric PRFs have the same values."""
    prf_1 = TessPRF(camera=2, ccd=3, shape=[20, 20], column=0, row=0)
    prf_2 = TessPRF(camera=2, ccd=3, shape=[10, 10], column=5, row=5)
    for c in [10, 8, 10, 7]:
        for r in [10, 10, 7, 7]:
            assert_allclose(prf_2(c, r, 1, 1, 1, 0),
                            prf_1(c, r, 1, 1, 1, 0)[5:15, 5:15],
                            rtol=1e-5)


@pytest.mark.remote_data
def test_kepler_prf_interpolation_consistency():
    """Ensures that the interpolated prf is consistent with calibration files.
    """
    prf = KeplerPRF(channel=56, shape=[15, 15], column=0, row=0)
    cal_prf = fits.open("http://archive.stsci.edu/missions/kepler/fpc/prf/"
                        "extracted/kplr16.4_2011265_prf.fits")
    cal_prf_subsampled = cal_prf[-1].data[25::50, 25::50]
    cal_prf_subsampled_normalized = cal_prf_subsampled / (cal_prf[-1].data.sum() * 0.02 ** 2)
    prf_data = prf(center_col=7.5, center_row=7.5, flux=1, scale_col=1, scale_row=1, rotation_angle=0)
    np.isclose(np.sum(np.abs(prf_data - cal_prf_subsampled_normalized)), 0)


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


@pytest.mark.parametrize("param_to_test", [("center_col"), ("center_row"), ("flux"),
                                           ("scale_col"), ("scale_row"), ("rotation_angle")])
def test_keplerprf_gradient_against_calculus(param_to_test):
    """is the gradient of KeplerPRF consistent with Calculus?
    """
    params = OrderedDict([('center_col', 7), ('center_row', 7), ('flux', 1000.),
                          ('scale_col', 1.), ('scale_row', 1.), ('rotation_angle', 0)])
    param_order = OrderedDict(zip(params.keys(), range(0, 6)))
    kwargs = {'channel': 56, 'shape': [15, 15], 'column': 0, 'row': 0}

    prf = KeplerPRF(**kwargs)
    h = 1e-8
    f = prf.evaluate
    inc_params = params.copy()
    # increment the parameter under test for later finite difference computation
    inc_params[param_to_test] += h
    # compute finite differences
    diff_prf = (f(**inc_params) - f(**params)) / h
    # compute analytical gradient
    prf_grad = prf.gradient(**params)
    # assert that the average absolute/relative error is less than 1e-5
    assert np.mean(np.abs(prf_grad[param_order[param_to_test]] - diff_prf) / (1. + np.abs(diff_prf))) < 1e-5


def test_gaussianprf_normalization():
    prf = GaussianPRF(shape=(20, 20), column=0, row=0)
    for flux in [0.1, 1.0, 10.0]:
        assert np.isclose(prf.evaluate(center_col=10, center_row=10, flux=flux).sum(), flux, rtol=1e-6)
        assert np.isclose(prf.evaluate(center_col=20, center_row=10, flux=flux).sum(), flux / 2, rtol=1e-6)
