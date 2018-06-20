from __future__ import division, print_function

import math

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_allclose
from oktopus import GaussianPrior, JointPrior, PoissonPosterior, UniformPrior
import pytest
from scipy.stats import mode

from ..prf import KeplerPRF, SimpleKeplerPRF
from ..prf_photometry import PRFPhotometry, SceneModel
from ..targetpixelfile import KeplerTargetPixelFile


def test_prf_normalization():
    """Does the PRF model integrate to the requested flux across the focal plane?"""
    for channel in [1, 20, 40, 60, 84]:
        for col in [123, 678]:
            for row in [234, 789]:
                shape = (18, 14)
                flux = 100
                prf = KeplerPRF(channel=channel, column=col, row=row, shape=shape)
                prf_sum = prf.evaluate(flux, col + shape[0]/2, row + shape[1]/2, 1, 1, 0).sum()
                assert np.isclose(prf_sum, flux, rtol=0.1)


"""
def test_prf_vs_aperture_photometry():
    # Is the PRF photometry result consistent with simple aperture photometry?
    tpf_fn = get_pkg_data_filename("data/ktwo201907706-c01-first-cadence.fits.gz")
    tpf = fits.open(tpf_fn)
    col, row = 173, 526
    prf = KeplerPRF(channel=tpf[0].header['CHANNEL'],
                    column=col, row=row,
                    shape=tpf[1].data.shape)
    scene = SceneModel(prfmodel=prf)
    fluxo, colo, rowo, _ = get_initial_guesses(data=tpf[1].data,
                                                 ref_col=prf.col_coord[0],
                                                 ref_row=prf.row_coord[0])
    bkg = mode(tpf[1].data, None)[0]
    prior = JointPrior(UniformPrior(lb=0.1*fluxo, ub=fluxo),
                       UniformPrior(lb=prf.col_coord[0], ub=prf.col_coord[-1]),
                       UniformPrior(lb=prf.row_coord[0], ub=prf.row_coord[-1]),
                       GaussianPrior(mean=1, var=1e-2),
                       GaussianPrior(mean=1, var=1e-2),
                       GaussianPrior(mean=0, var=1e-2),
                       UniformPrior(lb=bkg - .5*bkg, ub=bkg + .5*bkg))
    logL = PoissonPosterior(tpf[1].data, mean=scene, prior=prior)
    result = logL.fit(x0=prior.mean, method='powell')
    prf_flux, prf_col, prf_row, prf_scale_col, prf_scale_row, prf_rotation, prf_bkg = logL.opt_result.x
    assert result.success is True
    assert np.isclose(prf_col, colo, rtol=1e-1)
    assert np.isclose(prf_row, rowo, rtol=1e-1)
    assert np.isclose(prf_bkg, np.percentile(tpf[1].data, 10), rtol=0.1)

    # Test KeplerPRFPhotometry class
    kepler_phot = PRFPhotometry(scene_model=scene, prior=prior)
    tpf_flux = tpf[1].data.reshape((1, tpf[1].data.shape[0], tpf[1].data.shape[1]))
    kepler_phot.fit(tpf_flux=tpf_flux)
    opt_params = kepler_phot.opt_params.reshape(-1)
    assert np.isclose(opt_params[0], prf_flux, rtol=0.1)
    assert np.isclose(opt_params[1], prf_col, rtol=1e-1)
    assert np.isclose(opt_params[2], prf_row, rtol=1e-1)
    assert np.isclose(opt_params[-1], prf_bkg, rtol=0.1)
"""

"""
def test_get_initial_guesses():
    prf = SimpleKeplerPRF(channel=41, column=50, row=30, shape=[11, 11])
    prf_data = prf(flux=1, center_col=55.5, center_row=35.5)
    flux, col, row, _ = get_initial_guesses(prf_data, 50, 30)
    result = [flux, col, row]
    answer = [1, 55.5, 35.5]
    assert_allclose(result, answer, rtol=1e-1)
"""

def test_simple_kepler_prf():
    """Ensures that concentric PRFs have the same values.
    """

    prf_1 = SimpleKeplerPRF(channel=16, shape=[20, 20], column=0, row=0)
    prf_2 = SimpleKeplerPRF(channel=16, shape=[10, 10], column=5, row=5)
    for c in [10, 8, 10, 7]:
        for r in [10, 10, 7, 7]:
            assert_allclose(prf_2(flux=1, center_col=c, center_row=r),
                            prf_1(flux=1, center_col=c, center_row=r)[5:15, 5:15],
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
    sprf_data = sprf(flux=1, center_col=7.5, center_row=7.5)
    np.isclose(np.sum(np.abs(sprf_data - cal_prf_subsampled_normalized)), 0)


def test_scene_model():
    prf = SimpleKeplerPRF(channel=16, shape=[10, 10], column=15, row=15)
    scene = SceneModel(prfmodel=prf)
    assert scene.prfmodel.channel == 16


def test_get_model_prf():
    tpf_fn = get_pkg_data_filename("data/test-tpf-star.fits")
    tpf = KeplerTargetPixelFile(tpf_fn)

    prf = KeplerPRF(channel=tpf.channel, shape=tpf.shape[1:],
                    column=tpf.column, row=tpf.row)
    prf_from_tpf = tpf.get_prf_model()

    assert type(prf) == type(prf_from_tpf)
    assert prf.channel == prf_from_tpf.channel
    assert prf.shape == prf_from_tpf.shape
    assert prf.column == prf_from_tpf.column
    assert prf.row == prf_from_tpf.row
