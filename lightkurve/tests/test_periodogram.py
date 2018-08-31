import pytest
from astropy import units as u
import numpy as np
from numpy.testing import assert_array_equal
from ..periodogram import *
from ..lightcurvefile import KeplerLightCurveFile
from ..targetpixelfile import KeplerTargetPixelFile

TABBY_Q8 = ("https://archive.stsci.edu/missions/kepler/lightcurves"
            "/0084/008462852/kplr008462852-2011073133259_llc.fits")

@pytest.mark.remote_data
def test_periodogram():
    """Sanity check to verify that periodogram functions works"""
    ID = "09775454"
    tpf = KeplerTargetPixelFile.from_archive(ID, quarter=0,  quality_bitmask='hardest')
    lc =  tpf.to_lightcurve().normalize().remove_nans().remove_outliers()
    clc = lc.correct().remove_outliers().fill_gaps()
    p = clc.periodogram()

    numax = p.estimate_numax()
    dnu = p.estimate_delta_nu(numax)

    m, r, rho = p.estimate_stellar_parameters(numax, dnu, 5000)

    #test that calling stellar parameters works outside of the class
    assert(m == estimate_mass(numax, dnu, 5000))
    assert(r == estimate_radius(numax, dnu, 5000))
    assert(rho == estimate_mean_density(m,r))

@pytest.mark.remote_data
def test_from_lightcurve():
    """Can we create a Periodogram using `from_lightcurve`?"""
    lc = KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX
    Periodogram.from_lightcurve(lc=lc)
    # Can we provide frequencies as a list?
    frequencies = [0.1, 0.2, 0.3]
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies.value, frequencies)
    assert(pg.frequencies.unit == u.microhertz)
    # Can we provide frequencies as a Quantity in uHz?
    frequencies = np.array(frequencies) * u.microhertz
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies, frequencies)
    # Can we provide frequencies as a Quantity in Hz?
    frequencies = np.array(frequencies) * u.hertz
    pg = Periodogram.from_lightcurve(lc=lc, frequencies=frequencies)
    assert_array_equal(pg.frequencies, frequencies)


@pytest.mark.remote_data
def test_lightcurve_seismology_plot():
    """Sanity check to verify that periodogram plotting works"""
    KeplerLightCurveFile(TABBY_Q8).PDCSAP_FLUX.periodogram().plot()
