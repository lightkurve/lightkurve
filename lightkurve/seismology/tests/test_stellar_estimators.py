import pytest
from astropy import units as u
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from astropy import units as u
import astropy.constants as const
from uncertainties import ufloat

from ...lightcurve import LightCurve
from ...search import search_lightcurvefile
from .. import *
from ..stellar_estimators import (NUMAX_SOL, DELTANU_SOL, TEFF_SOL, G_SOL,
                                  estimate_radius, estimate_mass, estimate_logg)
import sys


bad_optional_imports = False
try:
    from astropy.stats.bls import BoxLeastSquares
except ImportError:
    bad_optional_imports = True


cM = ufloat(1.30, 0.09)
cR = ufloat(9.91, 0.24)
clogg = ufloat(2.559, 0.009)
ceteff = 80
cenumax = 0.75
cedeltanu = 0.012
cteff = 4531
cnumax = 46.12
cdeltanu = 4.934


def test_constants():
    """Assert the basic solar parameters are still loaded in and have
    appopriate units where necessary"""
    assert NUMAX_SOL.n == 3090.0
    assert NUMAX_SOL.s == 30.0
    assert DELTANU_SOL.n == 135.1
    assert DELTANU_SOL.s == 0.1
    assert TEFF_SOL.n == 5772.
    assert TEFF_SOL.s == 0.8
    assert np.isclose(G_SOL.value, 27420)
    assert G_SOL.unit == u.cm/u.second**2


def test_estimate_radius_basic():
    """Assert the basic functions of estimate_radius
    """
    R = estimate_radius(cnumax, cdeltanu, cteff)

    #Check units
    assert(R.unit == u.solRad)

    # Check returns right answer
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    # Check units on parameters
    R = estimate_radius(u.Quantity(cnumax, u.microhertz), cdeltanu, cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    R = estimate_radius(cnumax, u.Quantity(cdeltanu, u.microhertz), cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    R = estimate_radius(cnumax, cdeltanu, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    #Check works with a random selection of appropriate units
    R = estimate_radius(u.Quantity(cnumax, u.microhertz).to(1/u.day),
                             u.Quantity(cdeltanu, u.microhertz).to(u.hertz),
                             cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))


def test_estimate_radius_kwargs():
    """Test the kwargs of estimate_radius
    """
    R = estimate_radius(cnumax, cdeltanu, cteff, cenumax, cedeltanu, ceteff)
    assert R.error is not None

    #Check conditions for return
    t = estimate_radius(cnumax, cdeltanu, cteff, cenumax, cedeltanu)
    assert t.error is not None
    t = estimate_radius(cnumax, cdeltanu, cteff, cenumax, cedeltanu, ceteff)
    assert t.error is not None

    #Check units
    assert R.unit == u.solRad
    assert R.error.unit == u.solRad

    # Check returns right answer
    assert(np.isclose(R.value, cR.n, atol=cR.s))
    assert(np.isclose(R.error.value, cR.s, atol=.1))

    # Check units on parameters
    R = estimate_radius(cnumax, cdeltanu, cteff,
                        u.Quantity(cenumax, u.microhertz), cedeltanu, ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(R.error.value, cR.s, atol=.1))

    R = estimate_radius(cnumax, cdeltanu, cteff,
                        cenumax, u.Quantity(cedeltanu, u.microhertz), ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(R.error.value, cR.s, atol=.1))

    R = estimate_radius(cnumax, cdeltanu, cteff,
                        cenumax, cedeltanu, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(R.error.value, cR.s, atol=.1))

    #Check works with a random selection of appropriate units
    R = estimate_radius(cnumax, cdeltanu, cteff,
                        u.Quantity(cenumax, u.microhertz).to(1/u.day),
                        u.Quantity(cedeltanu, u.microhertz).to(u.hertz),
                        ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(R.error.value, cR.s, atol=.1))


def test_estimate_mass_basic():
    """Assert the basic functions of estimate_mass
    """
    M = estimate_mass(cnumax, cdeltanu, cteff)
    assert(M.unit == u.solMass)  # Check units
    assert(np.isclose(M.value, cM.n, rtol=cM.s))  # Check right answer

    # Check units on parameters
    M = estimate_mass(u.Quantity(cnumax, u.microhertz), cdeltanu, cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    M = estimate_mass(cnumax, u.Quantity(cdeltanu, u.microhertz), cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    M = estimate_mass(cnumax, cdeltanu, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    # Check works with a random selection of appropriate units
    M = estimate_mass(u.Quantity(cnumax, u.microhertz).to(1/u.day),
                      u.Quantity(cdeltanu, u.microhertz).to(u.hertz),
                      cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))


def test_estimate_mass_kwargs():
    """Test the kwargs of estimate_mass."""
    M = estimate_mass(cnumax, cdeltanu, cteff, cenumax, cedeltanu, ceteff)

    # Check units
    assert M.unit == u.solMass
    assert M.error.unit == u.solMass

    # Check returns right answer
    assert(np.isclose(M.value, cM.n, atol=cM.s))
    assert(np.isclose(M.error.value, cM.s, atol=.1))

    # Check units on parameters
    M = estimate_mass(cnumax, cdeltanu, cteff,
                      u.Quantity(cenumax, u.microhertz), cedeltanu, ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(M.error.value, cM.s, atol=.1))

    M = estimate_mass(cnumax, cdeltanu, cteff,
                      cenumax, u.Quantity(cedeltanu, u.microhertz), ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(M.error.value, cM.s, atol=.1))

    M = estimate_mass(cnumax, cdeltanu, cteff,
                      cenumax, cedeltanu, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(M.error.value, cM.s, atol=.1))

    #Check works with a random selection of appropriate units
    M = estimate_mass(cnumax, cdeltanu, cteff,
                      u.Quantity(cenumax, u.microhertz).to(1/u.day),
                      u.Quantity(cedeltanu, u.microhertz).to(u.hertz),
                      ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(M.error.value, cM.s, atol=.1))


def test_estimate_logg_basic():
    """Assert basic functionality of estimate_logg."""
    logg = estimate_logg(cnumax, cteff)
    # Check units
    assert(logg.unit == u.dex)
    # Check returns right answer
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    # Check units on parameters
    logg = estimate_logg(u.Quantity(cnumax, u.microhertz), cteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    logg = estimate_logg(cnumax, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    # Check works with a random selection of appropriate units
    logg = estimate_logg(u.Quantity(cnumax, u.microhertz).to(1/u.day), cteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))


def test_estimate_logg_kwargs():
    """Test the kwargs of estimate_logg."""
    logg = estimate_logg(cnumax, cteff, cenumax, ceteff)

    # Check units
    assert logg.unit == u.dex
    assert logg.error.unit == u.dex

    # Check returns right answer
    assert(np.isclose(logg.value, clogg.n, atol=clogg.s))
    assert(np.isclose(logg.error.value, clogg.s, atol=.1))

    # Check units on parameters
    logg = estimate_logg(cnumax, cteff, u.Quantity(cenumax, u.microhertz), ceteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logg.error.value, clogg.s, atol=.1))

    logg = estimate_logg(cnumax, cteff, cenumax, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logg.error.value, clogg.s, atol=.1))

    #Check works with a random selection of appropriate units
    logg = estimate_logg(cnumax, cteff,
                         u.Quantity(cenumax, u.microhertz).to(1/u.day),
                         ceteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logg.error.value, clogg.s, atol=.1))
