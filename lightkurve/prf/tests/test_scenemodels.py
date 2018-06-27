from __future__ import division, print_function

import numpy as np

from ...prf import FixedValuePrior, GaussianPrior, UniformPrior
from ...prf import StarPrior, BackgroundPrior, FocusPrior, MotionPrior
from ...prf import SceneModel
from ...prf import KeplerPRF


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


def test_scenemodel():
    col, row, maxflux, bgflux = 1, 2, 3, 4
    shape = (7, 8)
    model = SceneModel(star_priors=[StarPrior(col=GaussianPrior(mean=col, var=2**2),
                                              row=GaussianPrior(mean=row, var=2**2),
                                              flux=UniformPrior(lb=0, ub=maxflux),
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
    # Test parameters
    params = model.initial_guesses()
    assert params.stars[0].col == col
    assert len(params.to_array()) == 4  # The model has 4 free parameters
    # Predict should return an image
    assert model.predict().shape == shape
    # Test __repr__
    assert 'TESTSTAR' in str(model)
