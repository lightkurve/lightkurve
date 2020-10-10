import pytest
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal, assert_array_equal

from astropy import units as u
from astropy.time import Time
from astropy.stats.bls import BoxLeastSquares

from ..lightcurve import LightCurve
from ..periodogram import Periodogram
from ..utils import LightkurveWarning
import sys


def test_periodogram_basics():
    """Sanity check to verify that periodogram plotting works"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    lc = lc.normalize()
    pg = lc.to_periodogram()
    pg.plot()
    plt.close()
    pg.plot(view='period')
    plt.close()
    pg.show_properties()
    pg.to_table()
    str(pg)
    lc[400:500] = np.nan
    pg = lc.to_periodogram()


def test_periodogram_normalization():
    """Tests the normalization options"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1, flux_unit='electron/second')
    # Test amplitude normalization and correct units
    pg = lc.to_periodogram(normalization='amplitude')
    assert pg.power.unit == u.electron / u.second
    pg = lc.normalize(unit='ppm').to_periodogram(normalization='amplitude')
    assert pg.power.unit == u.cds.ppm

    # Test PSD normalization and correct units
    pg = lc.to_periodogram(freq_unit=u.microhertz, normalization='psd')
    assert pg.power.unit ==  (u.electron/u.second)**2 / u.microhertz
    pg = lc.normalize(unit='ppm').to_periodogram(freq_unit=u.microhertz, normalization='psd')
    assert pg.power.unit == u.cds.ppm**2 / u.microhertz


def test_periodogram_warnings():
    """Tests if warnings are raised for non-normalized periodogram input"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    lc = lc.normalize(unit='ppm')
    # Test amplitude normalization and correct units
    pg = lc.to_periodogram(normalization='amplitude')
    assert pg.power.unit == u.cds.ppm
    pg = lc.to_periodogram(freq_unit=u.microhertz, normalization='psd')
    assert pg.power.unit == u.cds.ppm**2 / u.microhertz


def test_periodogram_units():
    """Tests whether periodogram has correct units"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1, flux_unit='electron/second')
    p = lc.to_periodogram(normalization='amplitude')
    # Has units
    assert hasattr(p.frequency, 'unit')

    # Has the correct units
    assert p.frequency.unit == 1./u.day
    assert p.power.unit == u.electron / u.second
    assert p.period.unit == u.day
    assert p.frequency_at_max_power.unit == 1./u.day
    assert p.max_power.unit == u.electron / u.second


def test_periodogram_can_find_periods():
    """Periodogram should recover the correct period"""
    # Light curve that is noisy
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    # Add a 100 day period signal
    lc.flux += np.sin((lc.time.value/float(lc.time.value.max())) * 20 * np.pi)
    lc = lc.normalize()
    p = lc.to_periodogram(normalization='amplitude')
    assert np.isclose(p.period_at_max_power.value, 100, rtol=1e-3)


def test_periodogram_slicing():
    """Tests whether periodograms can be sliced"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    lc = lc.normalize()
    p = lc.to_periodogram()
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
    """Test if you can assign periods and frequencies."""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000) + 0.1)
    periods = np.arange(1, 100) * u.day
    lc = lc.normalize()
    p = lc.to_periodogram(period=periods)
    # Get around the floating point error
    assert np.isclose(np.sum(periods - p.period).value, 0, rtol=1e-14)
    frequency = np.arange(1, 100) * u.Hz
    p = lc.to_periodogram(frequency=frequency)
    assert np.isclose(np.sum(frequency - p.frequency).value, 0, rtol=1e-14)


def test_bin():
    """Test if you can bin the periodogram."""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000) + 0.1)
    lc = lc.normalize()
    p = lc.to_periodogram()
    assert len(p.bin(binsize=10, method='mean').frequency) == len(p.frequency)//10
    assert len(p.bin(binsize=10, method='median').frequency) == len(p.frequency)//10


def test_smooth():
    """Test if you can smooth the periodogram and check any pitfalls
    """
    np.random.seed(42)
    lc = LightCurve(time=np.arange(1000),
                    flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    lc = lc.normalize()
    p = lc.to_periodogram(normalization='psd', freq_unit=u.microhertz)
    # Test boxkernel and logmedian methods
    assert all(p.smooth(method='boxkernel').frequency == p.frequency)
    assert all(p.smooth(method='logmedian').frequency == p.frequency)
    # Check output units
    assert p.smooth().power.unit == p.power.unit

    # Check logmedian smooth that the mean of the smoothed power should
    # be consistent with the mean of the power
    assert np.isclose(np.mean(p.smooth(method='logmedian').power.value),
                     np.mean(p.power.value), atol=0.05*np.mean(p.power.value))


    # Can't pass filter_width below 0.
    with pytest.raises(ValueError) as err:
        p.smooth(method='boxkernel', filter_width=-5.)
    # Can't pass a filter_width in the wrong units
    with pytest.raises(ValueError) as err:
        p.smooth(method='boxkernel', filter_width=5.*u.day)
    assert err.value.args[0] == 'the `filter_width` parameter must have frequency units.'

    # Can't (yet) use a periodogram with a non-evenly spaced frequencies
    with pytest.raises(ValueError) as err:
        p = np.arange(1, 100)
        p = lc.to_periodogram(period=p)
        p.smooth()

    # Check logmedian doesn't work if I give the filter width units
    with pytest.raises(ValueError) as err:
        p.smooth(method='logmedian',  filter_width=5.*u.day)



def test_flatten():
    npts = 10000
    np.random.seed(12069424)
    lc = LightCurve(time=np.arange(npts),
                    flux=np.random.normal(1, 0.1, npts),
                    flux_err=np.zeros(npts)+0.1)
    lc = lc.normalize()
    p = lc.to_periodogram(normalization='psd', freq_unit=1/u.day)

    # Check method returns equal frequency
    assert all(p.flatten(method='logmedian').frequency == p.frequency)
    assert all(p.flatten(method='boxkernel').frequency == p.frequency)

    # Check logmedian flatten of white noise returns mean of ~unity
    assert np.isclose(np.mean(p.flatten(method='logmedian').power.value), 1.0,
                      atol=0.05)

    # Check return trend works
    s, b = p.flatten(return_trend=True)
    assert all(b.power == p.smooth(method='logmedian', filter_width=0.01).power)
    assert all(s.power == p.flatten().power)
    str(s)
    s.plot()
    plt.close()

def test_index():
    """Test if you can mask out periodogram
    """
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    lc = lc.normalize()
    p = lc.to_periodogram()
    mask = (p.frequency > 0.1*(1/u.day)) & (p.frequency < 0.2*(1/u.day))
    assert len(p[mask].frequency) == mask.sum()


def test_bls(caplog):
    ''' Test that BLS periodogram works and gives reasonable errors
    '''
    lc = LightCurve(time=np.linspace(0, 10, 200), flux=np.random.normal(100, 0.1, 200),
                    flux_err=np.zeros(200)+0.1)

    # should be able to make a periodogram
    p = lc.to_periodogram(method='bls')
    keys = ['period', 'power', 'duration', 'transit_time', 'depth', 'snr']
    assert np.all([key in  dir(p) for key in keys])

    p.plot()
    plt.close()

    # we should be able to specify some keywords
    lc.to_periodogram(method='bls', minimum_period=0.2, duration=0.1, maximum_period=0.5)

    # Ridiculous BLS spectra should break.
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(method='bls', frequency_factor=0.00001)
        assert err.value.args[0] == ('`period` contains over 72000001 points.Periodogram is too large to evaluate. Consider setting `frequency_factor` to a higher value.')

    # Some errors should occur
    p.compute_stats()
    for record in caplog.records:
        assert record.levelname == 'WARNING'
    assert len(caplog.records) == 3
    assert 'No period specified.' in caplog.text

    # No more errors
    stats = p.compute_stats(1, 0.1, 0)
    assert len(caplog.records) == 3
    assert isinstance(stats, dict)

    # Some errors should occur
    p.get_transit_model()
    for record in caplog.records:
        assert record.levelname == 'WARNING'
    assert len(caplog.records) == 6
    assert 'No period specified.' in caplog.text

    model = p.get_transit_model(1, 0.1, 0)
    # No more errors
    assert len(caplog.records) == 6
    # Model is LC
    assert isinstance(model, LightCurve)
    # Model is otherwise identical to LC
    assert np.in1d(model.time, lc.time).all()
    assert np.in1d(lc.time, model.time).all()

    mask = p.get_transit_mask(1, 0.1, 0)
    assert isinstance(mask, np.ndarray)
    assert isinstance(mask[0], np.bool_)
    assert mask.sum() < (~mask).sum()

    assert isinstance(p.period_at_max_power, u.Quantity)
    assert isinstance(p.duration_at_max_power, u.Quantity)
    assert isinstance(p.transit_time_at_max_power, Time)
    assert isinstance(p.depth_at_max_power, u.Quantity)


def test_bls_period_recovery():
    """Can BLS Periodogram recover the period of a synthetic light curve?"""
    # Planet parameters
    period = 2.0
    transit_time = 0.5
    duration = 0.1
    depth = 0.2
    flux_err = 0.01

    # Create the synthetic light curve
    time = np.arange(0, 20, 0.02)
    flux = np.ones_like(time)
    transit_mask = np.abs((time-transit_time+0.5*period) % period-0.5*period) < 0.5*duration
    flux[transit_mask] = 1.0 - depth
    flux += flux_err * np.random.randn(len(time))
    synthetic_lc = LightCurve(time=time, flux=flux)

    # Can BLS recover the period?
    bls_period = synthetic_lc.to_periodogram("bls").period_at_max_power
    assert_almost_equal(bls_period.value, period, decimal=2)
    # Does it work if we inject a sneaky NaN?
    synthetic_lc.flux[10] = np.nan
    bls_period = synthetic_lc.to_periodogram("bls").period_at_max_power
    assert_almost_equal(bls_period.value, period, decimal=2)
    # Does it work if all errors are NaNs?
    # This is a regression test for issue #428
    synthetic_lc.flux_err = np.array([np.nan] * len(time))
    assert_almost_equal(bls_period.value, period, decimal=2)


def test_error_messages():
    """Test periodogram raises reasonable errors
    """
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)

    # Can't specify period range and frequency range
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(maximum_frequency=0.1, minimum_period=10)

    # Can't have a minimum frequency > maximum frequency
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(maximum_frequency=0.1, minimum_frequency=10)
    assert err.value.args[0] == 'minimum_frequency cannot be larger than maximum_frequency'

    # Can't have a minimum period > maximum period
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(maximum_period=0.1, minimum_period=10)
    assert err.value.args[0] == 'minimum_period cannot be larger than maximum_period'

    # Can't specify periods and frequencies
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(frequency=np.arange(10), period=np.arange(10))


    # No unitless periodograms
    with pytest.raises(ValueError) as err:
        Periodogram([0], [1])
    assert err.value.args[0] == 'frequency must be an `astropy.units.Quantity` object.'

    # No unitless periodograms
    with pytest.raises(ValueError) as err:
        Periodogram([0]*u.Hz, [1])
    assert err.value.args[0] == 'power must be an `astropy.units.Quantity` object.'

    # No single value periodograms
    with pytest.raises(ValueError) as err:
        Periodogram([0]*u.Hz, [1]*u.K)
    assert err.value.args[0] == 'frequency and power must have a length greater than 1.'

    # No uneven arrays
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2, 3]*u.Hz, [1, 1]*u.K)
    assert err.value.args[0] == 'frequency and power must have the same length.'

    # Bad frequency units
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2]*u.K, [1, 1, 1]*u.K)
    assert err.value.args[0] == 'Frequency must be in units of 1/time.'

    # Bad binning
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2]*u.Hz, [1, 1, 1]*u.K).bin(binsize=-2)
    assert err.value.args[0] == 'binsize must be larger than or equal to 1'

    # Bad binning method
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2]*u.Hz, [1, 1, 1]*u.K).bin(method='not-implemented')
    assert("method 'not-implemented' is not supported" in err.value.args[0])

    # Bad smooth method
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2]*u.Hz, [1, 1, 1]*u.K).smooth(method="not-implemented")
    assert("method 'not-implemented' is not supported" in err.value.args[0])


def test_bls_period():
    """Regression test for #514."""
    lc = LightCurve(time=[1, 2, 3], flux=[4, 5, 6])
    period = [1, 2, 3, 4, 5]
    pg = lc.to_periodogram(method="bls", period=period)
    assert_array_equal(pg.period.value, period)
    with pytest.raises(ValueError) as err:  # NaNs should raise a nice error message
        lc.to_periodogram(method="bls", period=[1, 2, 3, np.nan, 4])
    assert("period" in err.value.args[0])
