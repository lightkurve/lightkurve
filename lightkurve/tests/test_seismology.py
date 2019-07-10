import pytest
from astropy import units as u
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from astropy import units as u
import astropy.constants as const
from uncertainties import ufloat

from ..lightcurve import LightCurve
from ..search import search_lightcurvefile
from ..seismology import *
from ..seismology import NUMAX_SOL, DNU_SOL, TEFF_SOL, G_SOL
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
cednu = 0.012
cteff = 4531
cnumax = 46.12
cdnu = 4.934

def test_constants():
    """Assert the basic solar parameters are still loaded in and have
    appopriate units where necessary"""
    assert NUMAX_SOL.n == 3090.0
    assert NUMAX_SOL.s == 30.0
    assert DNU_SOL.n == 135.1
    assert DNU_SOL.s == 0.1
    assert TEFF_SOL.n == 5772.
    assert TEFF_SOL.s == 0.8
    assert np.isclose(G_SOL.value, 27420)
    assert G_SOL.unit == u.cm/u.second**2

def test_estimate_radius_basic():
    """Assert the basic functions of estimate_radius
    """
    R = estimate_radius(cnumax, cdnu, cteff)

    #Check units
    assert(R.unit == u.solRad)

    # Check returns right answer
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    # Check units on parameters
    R = estimate_radius(u.Quantity(cnumax, u.microhertz), cdnu, cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    R = estimate_radius(cnumax, u.Quantity(cdnu, u.microhertz), cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    R = estimate_radius(cnumax, cdnu, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

    #Check works with a random selection of appropriate units
    R = estimate_radius(u.Quantity(cnumax, u.microhertz).to(1/u.day),
                             u.Quantity(cdnu, u.microhertz).to(u.hertz),
                             cteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))

def test_estimate_radius_kwargs():
    """Test the kwargs of estimate_radius
    """
    R, Re = estimate_radius(cnumax, cdnu, cteff,
                                cenumax, cednu, ceteff)

    #Check conditions for return
    t = estimate_radius(cnumax, cdnu, cteff, cenumax, cednu)
    assert t.shape == ()
    t = estimate_radius(cnumax, cdnu, cteff, cenumax, cednu, ceteff)
    assert len(t) == 2

    #Check units
    assert R.unit == u.solRad
    assert Re.unit == u.solRad

    # Check returns right answer
    assert(np.isclose(R.value, cR.n, atol=cR.s))
    assert(np.isclose(Re.value, cR.s, atol=.1))

    # Check units on parameters
    R, Re = estimate_radius(cnumax, cdnu, cteff,
                        u.Quantity(cenumax, u.microhertz), cednu, ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(Re.value, cR.s, atol=.1))

    R, Re = estimate_radius(cnumax, cdnu, cteff,
                        cenumax, u.Quantity(cednu, u.microhertz), ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(Re.value, cR.s, atol=.1))

    R, Re = estimate_radius(cnumax, cdnu, cteff,
                        cenumax, cednu, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(Re.value, cR.s, atol=.1))

    #Check works with a random selection of appropriate units
    R, Re = estimate_radius(cnumax, cdnu, cteff,
                             u.Quantity(cenumax, u.microhertz).to(1/u.day),
                             u.Quantity(cednu, u.microhertz).to(u.hertz),
                             ceteff)
    assert(np.isclose(R.value, cR.n, rtol=cR.s))
    assert(np.isclose(Re.value, cR.s, atol=.1))

def test_estimate_mass_basic():
    """Assert the basic functions of estimate_mass
    """
    M = estimate_mass(cnumax, cdnu, cteff)

    #Check units
    assert(M.unit == u.solMass)

    # Check returns right answer
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    # Check units on parameters
    M = estimate_mass(u.Quantity(cnumax, u.microhertz), cdnu, cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    M = estimate_mass(cnumax, u.Quantity(cdnu, u.microhertz), cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    M = estimate_mass(cnumax, cdnu, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

    #Check works with a random selection of appropriate units
    M = estimate_mass(u.Quantity(cnumax, u.microhertz).to(1/u.day),
                             u.Quantity(cdnu, u.microhertz).to(u.hertz),
                             cteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))

def test_estimate_mass_kwargs():
    """Test the kwargs of estimate_mass
    """
    M, Me = estimate_mass(cnumax, cdnu, cteff,
                                cenumax, cednu, ceteff)

    #Check conditions for return
    t = estimate_mass(cnumax, cdnu, cteff, cenumax, cednu)
    assert t.shape == ()
    t = estimate_mass(cnumax, cdnu, cteff, cenumax, cednu, ceteff)
    assert len(t) == 2

    #Check units
    assert M.unit == u.solMass
    assert Me.unit == u.solMass

    # Check returns right answer
    assert(np.isclose(M.value, cM.n, atol=cM.s))
    assert(np.isclose(Me.value, cM.s, atol=.1))

    # Check units on parameters
    M, Me = estimate_mass(cnumax, cdnu, cteff,
                        u.Quantity(cenumax, u.microhertz), cednu, ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(Me.value, cM.s, atol=.1))

    M, Me = estimate_mass(cnumax, cdnu, cteff,
                        cenumax, u.Quantity(cednu, u.microhertz), ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(Me.value, cM.s, atol=.1))

    M, Me = estimate_mass(cnumax, cdnu, cteff,
                        cenumax, cednu, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(Me.value, cM.s, atol=.1))

    #Check works with a random selection of appropriate units
    M, Me = estimate_mass(cnumax, cdnu, cteff,
                             u.Quantity(cenumax, u.microhertz).to(1/u.day),
                             u.Quantity(cednu, u.microhertz).to(u.hertz),
                             ceteff)
    assert(np.isclose(M.value, cM.n, rtol=cM.s))
    assert(np.isclose(Me.value, cM.s, atol=.1))

def test_estimate_logg_basic():
    """Assert basic functionality of estimate_logg
    """
    logg = estimate_logg(cnumax, cteff)

    #Check units
    assert(logg.unit == u.dex)

    # Check returns right answer
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))

    # Check units on parameters
    logg = estimate_logg(u.Quantity(cnumax, u.microhertz), cteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))

    logg = estimate_logg(cnumax, u.Quantity(cteff, u.Kelvin))
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))

    #Check works with a random selection of appropriate units
    logg = estimate_logg(u.Quantity(cnumax, u.microhertz).to(1/u.day),
                             cteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))

def test_estimate_logg_kwargs():
    """Test the kwargs of estimate_logg
    """

    logg, logge = estimate_logg(cnumax, cteff,
                                cenumax, ceteff)

    #Check conditions for return
    t = estimate_logg(cnumax, cteff, cenumax)
    assert t.shape == ()
    t = estimate_logg(cnumax, cteff, cenumax, ceteff)
    assert len(t) == 2

    #Check units
    assert logg.unit == u.dex
    assert logge.unit == u.dex

    # Check returns right answer
    assert(np.isclose(logg.value, clogg.n, atol=clogg.s))
    assert(np.isclose(logge.value, clogg.s, atol=.1))

    # Check units on parameters
    logg, logge = estimate_logg(cnumax, cteff,
                        u.Quantity(cenumax, u.microhertz), ceteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logge.value, clogg.s, atol=.1))

    logg, logge = estimate_logg(cnumax, cteff,
                        cenumax, u.Quantity(ceteff, u.Kelvin))
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logge.value, clogg.s, atol=.1))

    #Check works with a random selection of appropriate units
    logg, logge = estimate_logg(cnumax, cteff,
                             u.Quantity(cenumax, u.microhertz).to(1/u.day),
                             ceteff)
    assert(np.isclose(logg.value, clogg.n, rtol=clogg.s))
    assert(np.isclose(logge.value, clogg.s, atol=.1))
