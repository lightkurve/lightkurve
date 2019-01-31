import pytest
from astropy import units as u
import numpy as np
from ..lightcurve import LightCurve
from ..periodogram import Periodogram
import sys

try:
    from astropy.stats.bls import BoxLeastSquares
except:
    print('no bls, tests will be skipped')

bad_optional_imports = np.any([('astropy.stats.bls' not in sys.modules)])


def test_periodogram_basics():
    """Sanity check to verify that periodogram plotting works"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    pg = lc.to_periodogram()
    pg.plot()
    pg.plot(view='period')
    pg.show_properties()
    pg.to_table()
    str(pg)


def test_periodogram_units():
    """Tests whether periodogram has correct units"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram()
    # Has units
    assert hasattr(p.frequency, 'unit')

    # Has the correct units
    assert p.frequency.unit == 1./u.day
    assert p.power.unit == u.cds.ppm**2*u.day
    assert p.period.unit == u.day
    assert p.frequency_at_max_power.unit == 1./u.day
    assert p.max_power.unit == u.cds.ppm**2*u.day


def test_periodogram_can_find_periods():
    """Periodogram should recover the correct period"""
    # Light curve that is noisy
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    # Add a 100 day period signal
    lc.flux *= np.sin((lc.time/float(lc.time.max())) * 20 * np.pi)
    p = lc.to_periodogram()
    assert np.isclose(p.period_at_max_power.value, 100, rtol=1e-3)


def test_periodogram_slicing():
    """Tests whether periodograms can be sliced"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
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
    p = lc.to_periodogram()
    assert len(p.bin(binsize=10, method='mean').frequency) == len(p.frequency)//10
    assert len(p.bin(binsize=10, method='median').frequency) == len(p.frequency)//10


def test_smooth():
    """Test if you can smooth the periodogram and check any pitfalls
    """
    lc = LightCurve(time=np.arange(1000),
                    flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram()
    # Test boxkernel and logmedian methods
    assert all(p.smooth(method='boxkernel').frequency == p.frequency)
    assert all(p.smooth(method='logmedian').frequency == p.frequency)
    # Check output units
    assert p.smooth().power.unit == p.power.unit

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
    lc = LightCurve(time=np.arange(1000),
                    flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram()

    # Check method returns equal frequency
    assert all(p.flatten(method='logmedian').frequency == p.frequency)
    assert all(p.flatten(method='boxkernel').frequency == p.frequency)

    # Check return trend works
    s, b = p.flatten(return_trend=True)
    assert all(b.power == p.smooth(method='logmedian', filter_width=0.01).power)
    assert all(s.power == p.flatten().power)
    str(s)
    s.plot()


def test_index():
    """Test if you can mask out periodogram
    """
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram()
    mask = (p.frequency > 0.1*(1/u.day)) & (p.frequency < 0.2*(1/u.day))
    assert len(p[mask].frequency) == mask.sum()

@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_bls(caplog):
    ''' Test that BLS periodogram works and gives reasonable errors
    '''
    lc = LightCurve(time=np.linspace(0, 10, 1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)

    # should be able to make a periodogram
    p = lc.to_periodogram(method='bls')
    keys = ['period', 'power', 'duration', 'transit_time', 'depth', 'snr']
    assert np.all([key in  dir(p) for key in keys])

    p.plot()

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
    assert len(caplog.records) == 4
    assert 'No period specified.' in caplog.text

    # No more errors
    stats = p.compute_stats(1, 0.1, 0)
    assert len(caplog.records) == 4
    assert isinstance(stats, dict)

    # Some errors should occur
    p.get_transit_model()
    for record in caplog.records:
        assert record.levelname == 'WARNING'
    assert len(caplog.records) == 7
    assert 'No period specified.' in caplog.text

    model = p.get_transit_model(1, 0.1, 0)
    # No more errors
    assert len(caplog.records) == 7
    # Model is LC
    assert isinstance(model, LightCurve)
    # Model is otherwise identical to LC
    assert np.in1d(model.time, lc.time).all()
    assert np.in1d(lc.time, model.time).all()

    mask = p.get_transit_mask(1, 0.1, 0)
    assert isinstance(mask, np.ndarray)
    assert isinstance(mask[0], np.bool_)
    assert mask.sum() > (~mask).sum()

    assert isinstance(p.period_at_max_power, u.Quantity)
    assert isinstance(p.duration_at_max_power, float)
    assert isinstance(p.transit_time_at_max_power, float)
    assert isinstance(p.depth_at_max_power, float)


def test_error_messages():
    """Test periodogram raises reasonable errors
    """
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)

    # Can't specify period range and frequency range
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(max_frequency=0.1, min_period=10)

    # Can't have a minimum frequency > maximum frequency
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(max_frequency=0.1, min_frequency=10)
        assert err.value.args[0] == 'min_frequency cannot be larger than max_frequency'

    # Can't have a minimum period > maximum period
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(max_period=0.1, min_period=10)
        assert err.value.args[0] == 'min_period cannot be larger than max_period'

    # Can't specify periods and frequencies
    with pytest.raises(ValueError) as err:
        lc.to_periodogram(frequency=np.arange(10), period=np.arange(10))

    # Don't accept NaNs
    with pytest.raises(ValueError) as err:
        lc_with_nans = lc.copy()
        lc_with_nans.flux[0] = np.nan
        lc_with_nans.to_periodogram()
        assert('Lightcurve contains NaN values.' in err.value.args[0])

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
        assert("is not a valid method, must be 'mean' or 'median'" in err.value.args[0])

    # Bad smooth method
    with pytest.raises(ValueError) as err:
        Periodogram([0, 1, 2]*u.Hz, [1, 1, 1]*u.K).smooth(method="not-implemented")
        assert("parameter must be one 'boxkernel' or 'logmedian'" in err.value.args[0])
