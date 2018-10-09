import pytest
from astropy import units as u
import numpy as np
from numpy.testing import assert_array_equal
from ..periodogram import *
from ..lightcurvefile import KeplerLightCurveFile
from ..lightcurve import LightCurve
from ..targetpixelfile import KeplerTargetPixelFile
from ..periodogram import Periodogram

def test_lightcurve_seismology_plot():
    """Sanity check to verify that periodogram plotting works"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    lc.periodogram().plot()

def test_periodogram_units():
    """Tests whether periodogram has correct units"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    p = lc.periodogram()
    # Has units
    assert hasattr(p.frequency, 'unit')

    # Has the correct units
    assert p.frequency.unit == 1./u.day
    assert p.power.unit == u.cds.ppm**2*u.day
    assert p.period.unit == u.day
    assert p.frequency_at_max_power.unit == 1./u.day

def test_periodogram_can_find_periods():
    """Periodogram should recover the correct period"""
    # Light curve that is noisy
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    # Add a 100 day period signal
    lc.flux *= np.sin((lc.time/float(lc.time.max())) * 20 * np.pi)
    p = lc.periodogram()
    assert np.isclose(p.period_at_max_power.value, 100, rtol=1e-3)

def test_periodogram_slicing():
    """Tests whether periodograms can be sliced"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    p = lc.periodogram()
    assert len(p[0:200].frequency) == 200

    # Test divide
    orig = p.power.sum()
    p /= 2
    assert np.sum(p.power) == orig/2

    # Test multiplication
    p *= 0
    assert np.sum(p.power) == 0

    # Test addition
    p += 100
    assert np.all(p.power.value >= 100)

    # Test subtraction
    p -= 100
    assert np.sum(p.power) == 0

def test_assign_periods():
    ''' Test if you can assign periods and frequencies
    '''
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    periods = np.arange(0, 100) * u.day
    p = lc.periodogram(period=periods)
    # Get around the floating point error
    assert np.isclose(np.sum(periods - p.period).value, 0, rtol=1e-14)
    frequency = np.arange(0, 100) * u.Hz
    p = lc.periodogram(frequency=frequency)
    assert np.isclose(np.sum(frequency - p.frequency).value, 0, rtol=1e-14)

def test_bin():
    ''' Test if you can bin the periodogram
    '''
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    p = lc.periodogram()
    assert len(p.bin(binsize=10).frequency) == len(p.frequency)//10

def test_index():
    '''Test if you can mask out periodogram
    '''
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)
    p = lc.periodogram()
    mask = (p.frequency > 0.1*(1/u.day)) & (p.frequency < 0.2*(1/u.day))
    assert len(p[mask].frequency) == mask.sum()

def test_error_messages():
    '''Test periodogram raises reasonable errors
    '''
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000), flux_err=np.zeros(1000)+0.1)

    # Can't specify period range and frequency range
    with pytest.raises(ValueError) as err:
        lc.periodogram(max_frequency=0.1, min_period=10)

    # Can't have a minimum frequency > maximum frequency
    with pytest.raises(ValueError) as err:
        lc.periodogram(max_frequency=0.1, min_frequency=10)

    # Can't specify periods and frequencies
    with pytest.raises(ValueError) as err:
        lc.periodogram(frequency=np.arange(10), period=np.arange(10))

    # No unitless periodograms
    with pytest.raises(ValueError) as err:
        Periodogram([0], [1])
    assert err.value.args[0] == 'Power must have units.'

    # No single value periodograms
    with pytest.raises(ValueError) as err:
        Periodogram([0]*u.Hz, [1]*u.K)
    assert err.value.args[0] == 'Frequency and power must have a length greater than 1.'

    # No uneven arrays
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2, 3]*u.Hz, [1, 1]*u.K)
    assert err.value.args[0] == 'Frequency and power must be the same length.'

    # Bad frequency units
    with pytest.raises(ValueError) as err:
        Periodogram([0,1,2]*u.K, [1,1,1]*u.K)
    assert err.value.args[0] == 'Frequency must be in units of 1/time.'
