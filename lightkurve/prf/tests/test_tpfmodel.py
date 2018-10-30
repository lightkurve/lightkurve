"""Test the features of the lightkurve.prf.tpfmodels module."""
from __future__ import division, print_function

import os
import pytest

from astropy.io import fits
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import mode

from ... import PACKAGEDIR
from ...prf import FixedValuePrior, GaussianPrior, UniformPrior
from ...prf import StarPrior, BackgroundPrior, FocusPrior, MotionPrior
from ...prf import TPFModel, PRFPhotometry
from ...prf import SimpleKeplerPRF, KeplerPRF


def test_fixedvalueprior():
    fvp = FixedValuePrior(1.5)
    assert fvp.mean == 1.5
    assert fvp(1.5) == 0


def test_starprior():
    """Tests the StarPrior class."""
    col, row, flux = 1, 2, 3
    sp = StarPrior(col=GaussianPrior(mean=col, var=0.1),
                   row=GaussianPrior(mean=row, var=0.1),
                   flux=GaussianPrior(mean=flux, var=0.1))
    assert sp.col.mean == col
    assert sp.row.mean == row
    assert sp.flux.mean == flux
    assert sp.evaluate(col, row, flux) == 0
    # The object should be callable
    assert sp(col, row, flux + 0.1) == sp.evaluate(col, row, flux + 0.1)
    # A point further away from the mean should have a larger negative log likelihood
    assert sp.evaluate(col, row, flux) < sp.evaluate(col, row, flux + 0.1)
    # Object should have a nice __repr__
    assert 'StarPrior' in str(sp)


def test_backgroundprior():
    """Tests the BackgroundPrior class."""
    flux = 2.
    bp = BackgroundPrior(flux=flux)
    assert bp.flux.mean == flux
    assert bp(flux) == 0.
    assert not np.isfinite(bp(flux + 0.1))


def test_tpf_model_simple():
    prf = SimpleKeplerPRF(channel=16, shape=[10, 10], column=15, row=15)
    model = TPFModel(prfmodel=prf)
    assert model.prfmodel.channel == 16


def test_tpf_model():
    col, row, flux, bgflux = 1, 2, 3, 4
    shape = (7, 8)
    model = TPFModel(star_priors=[StarPrior(col=GaussianPrior(mean=col, var=2**2),
                                            row=GaussianPrior(mean=row, var=2**2),
                                            flux=UniformPrior(lb=flux - 0.5, ub=flux + 0.5),
                                            targetid="TESTSTAR")],
                     background_prior=BackgroundPrior(flux=GaussianPrior(mean=bgflux, var=bgflux)),
                     focus_prior=FocusPrior(scale_col=GaussianPrior(mean=1, var=0.0001),
                                            scale_row=GaussianPrior(mean=1, var=0.0001),
                                            rotation_angle=UniformPrior(lb=-3.1415, ub=3.1415)),
                     motion_prior=MotionPrior(shift_col=GaussianPrior(mean=0., var=0.01),
                                              shift_row=GaussianPrior(mean=0., var=0.01)),
                     prfmodel=KeplerPRF(channel=40, shape=shape, column=30, row=20),
                     fit_background=True,
                     fit_focus=False,
                     fit_motion=False)
    # Sanity checks
    assert model.star_priors[0].col.mean == col
    assert model.star_priors[0].targetid == 'TESTSTAR'
    # Test initial guesses
    params = model.get_initial_guesses()
    assert params.stars[0].col == col
    assert params.stars[0].row == row
    assert params.stars[0].flux == flux
    assert params.background.flux == bgflux
    assert len(params.to_array()) == 4  # The model has 4 free parameters
    assert_allclose([col, row, flux, bgflux], params.to_array(), rtol=1e-5)
    # Predict should return an image
    assert model.predict().shape == shape
    # Test __repr__
    assert 'TESTSTAR' in str(model)


# Tagging the test below as `remote_data` because AppVeyor hangs on this test;
# at present we don't understand why.
@pytest.mark.remote_data
def test_tpf_model_fitting():
    # Is the PRF photometry result consistent with simple aperture photometry?
    tpf_fn = os.path.join(PACKAGEDIR, "tests", "data", "ktwo201907706-c01-first-cadence.fits.gz")
    tpf = fits.open(tpf_fn)
    col, row = 173, 526
    fluxsum = np.sum(tpf[1].data)
    bkg = mode(tpf[1].data, None)[0]
    prfmodel = KeplerPRF(channel=tpf[0].header['CHANNEL'],
                         column=col, row=row,
                         shape=tpf[1].data.shape)
    star_priors = [StarPrior(col=UniformPrior(lb=prfmodel.col_coord[0], ub=prfmodel.col_coord[-1]),
                             row=UniformPrior(lb=prfmodel.row_coord[0], ub=prfmodel.row_coord[-1]),
                             flux=UniformPrior(lb=0.5*fluxsum, ub=1.5*fluxsum))]
    background_prior = BackgroundPrior(flux=UniformPrior(lb=.5*bkg, ub=1.5*bkg))
    model = TPFModel(star_priors=star_priors,
                     background_prior=background_prior,
                     prfmodel=prfmodel)
    # Does fitting run without errors?
    result = model.fit(tpf[1].data)
    # Can we change model parameters?
    assert result.motion.fitted == False
    model.fit_motion = True
    result = model.fit(tpf[1].data)
    assert result.motion.fitted == True
    # Does fitting via the PRFPhotometry class run without errors?
    phot = PRFPhotometry(model)
    phot.run([tpf[1].data])


def test_empty_model():
    """Can we fit the background flux in a model without stars?"""
    shape = (4, 3)
    bgflux = 1.23
    background_prior = BackgroundPrior(flux=UniformPrior(lb=0, ub=10))
    model = TPFModel(background_prior=background_prior, fit_background=True)
    background = bgflux * np.ones(shape=shape)
    results = model.fit(background)
    assert np.isclose(results.background.flux, bgflux, rtol=1e-2)


def test_model_with_one_star():
    """Can we fit the background flux in a model with one star?"""
    channel = 42
    shape = (10, 12)
    starflux, col, row = 1000., 60., 70.
    bgflux = 10.
    scale_col, scale_row, rotation_angle = 1.2, 1.3, 0.2
    prf = KeplerPRF(channel=channel, shape=shape, column=col, row=row)
    star_prior = StarPrior(col=GaussianPrior(col + 6, 0.01),
                           row=GaussianPrior(row + 6, 0.01),
                           flux=UniformPrior(lb=0.5*starflux, ub=1.5*starflux))
    background_prior = BackgroundPrior(flux=UniformPrior(lb=0, ub=100))
    focus_prior = FocusPrior(scale_col=UniformPrior(lb=0.5, ub=1.5),
                             scale_row=UniformPrior(lb=0.5, ub=1.5),
                             rotation_angle=UniformPrior(lb=0., ub=0.5))
    model = TPFModel(star_priors=[star_prior],
                     background_prior=background_prior,
                     focus_prior=focus_prior,
                     prfmodel=prf,
                     fit_background=True,
                     fit_focus=True)
    # Generate and fit fake data
    fake_data = bgflux + prf(col + 6, row + 6, starflux,
                             scale_col=scale_col, scale_row=scale_row,
                             rotation_angle=rotation_angle)
    results = model.fit(fake_data, tol=1e-12, options={'maxiter': 100})
    # Do the results match the input?
    assert np.isclose(results.stars[0].col, col + 6)
    assert np.isclose(results.stars[0].row, row + 6)
    assert np.isclose(results.stars[0].flux, starflux)
    assert np.isclose(results.background.flux, bgflux)
    assert np.isclose(results.focus.scale_col, scale_col)
    assert np.isclose(results.focus.scale_row, scale_row)
    assert np.isclose(results.focus.rotation_angle, rotation_angle)
