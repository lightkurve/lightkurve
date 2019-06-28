import pytest
from astropy import units as u
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.signal import unit_impulse as deltafn

from ..lightcurve import LightCurve
from ..search import search_lightcurvefile
from ..periodogram import Periodogram
from ..periodogram import SNRPeriodogram
import sys


bad_optional_imports = False
try:
    from astropy.stats.bls import BoxLeastSquares
except ImportError:
    bad_optional_imports = True

def test_asteroseismology():
    datalist = search_lightcurvefile('KIC11615890')
    data = datalist.download_all()
    lc = data[0].PDCSAP_FLUX.normalize().flatten()
    for nlc in data[0:5]:
        lc = lc.append(nlc.PDCSAP_FLUX.normalize().flatten())
    lc = lc.remove_nans()
    pg = lc.to_periodogram(normalization='psd')
    snr = pg.flatten()
    snr.estimate_numax()

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

def test_periodogram_normalization():
    """Tests the normalization options"""
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    # Test amplitude normalization and correct units
    pg = lc.to_periodogram(normalization='amplitude')
    assert pg.power.unit == u.cds.ppm

    # Test PSD normalization and correct units
    pg = lc.to_periodogram(freq_unit=u.microhertz, normalization='psd')
    assert pg.power.unit == u.cds.ppm**2 / u.microhertz

def test_periodogram_units():
    """Tests whether periodogram has correct units"""
    # Fake, noisy data
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram(normalization='amplitude')
    # Has units
    assert hasattr(p.frequency, 'unit')

    # Has the correct units
    assert p.frequency.unit == 1./u.day
    assert p.power.unit == u.cds.ppm
    assert p.period.unit == u.day
    assert p.frequency_at_max_power.unit == 1./u.day
    assert p.max_power.unit == u.cds.ppm


def test_periodogram_can_find_periods():
    """Periodogram should recover the correct period"""
    # Light curve that is noisy
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    # Add a 100 day period signal
    lc.flux *= np.sin((lc.time/float(lc.time.max())) * 20 * np.pi)
    p = lc.to_periodogram(normalization='amplitude')
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
    np.random.seed(42)
    lc = LightCurve(time=np.arange(1000),
                    flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
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

def test_index():
    """Test if you can mask out periodogram
    """
    lc = LightCurve(time=np.arange(1000), flux=np.random.normal(1, 0.1, 1000),
                    flux_err=np.zeros(1000)+0.1)
    p = lc.to_periodogram()
    mask = (p.frequency > 0.1*(1/u.day)) & (p.frequency < 0.2*(1/u.day))
    assert len(p[mask].frequency) == mask.sum()

def generate_test_spectrum():
    """Generates a simple solar-like oscillator spectrum of oscillation modes
    """
    f = np.arange(0, 4000., 0.4)
    p = np.ones(len(f))
    nmx = 2500.
    fs = f.max()/len(f)

    s = 0.25*nmx/2.335    #std of the hump
    p *= 10 * np.exp(-0.5*(f-nmx)**2/s**2)  #gaussian profile of the hump

    m = np.zeros(len(f))
    lo = int(np.floor(.5*nmx/fs))
    hi = int(np.floor(1.5*nmx/fs))

    dnu_true = 0.294 * nmx ** 0.772
    modelocs = np.arange(lo, hi, dnu_true/2, dtype=int)

    for modeloc in modelocs:
        m += deltafn(len(f), modeloc)
    p *= m
    p += 1
    return f, p, nmx, dnu_true

def test_estimate_numax_basics():
    """Test if we can estimate a numax
    """
    f, p, true_numax, _ = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    numax = snr.estimate_numax()

    #Assert recovers numax within 10%
    assert(np.isclose(true_numax, numax.value, atol=.1*true_numax))
    #Assert numax has unit equal to input frequency unit
    assert(numax.unit == u.microhertz)
    #Assert numax estimator works when input frequency is not in microhertz
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    snr = SNRPeriodogram(fday, u.Quantity(p, None))
    numax = snr.estimate_numax()
    nmxday = u.Quantity(true_numax*u.microhertz, 1/u.day)
    assert(np.isclose(nmxday, numax, atol=.1*nmxday))

def test_estimate_numax_kwargs():
    """Test if we can estimate a numax using its various keyword arguments
    """
    f, p, true_numax, _ = generate_test_spectrum()
    std = 0.25*true_numax/2.335  # The standard deviation of the mode envelope
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    numaxs = np.linspace(true_numax-2*std, true_numax+2*std, 500)
    numax = snr.estimate_numax(numaxs=numaxs)

    #Assert we can recover numax using a custom numax
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))

    #Assert we can't pass custom numaxs outside a functional range
    with pytest.raises(ValueError) as err:
        numax = snr.estimate_numax(numaxs=np.linspace(-5, 5.))
    with pytest.raises(ValueError) as err:
        numax = snr.estimate_numax(numaxs=np.linspace(1., 5000.))

    #Assert it doesn't matter what units of frqeuency numaxs are passed in as
    #Assert the output is still in the same units as the object frequencies
    daynumaxs = u.Quantity(numaxs*u.microhertz, 1/u.day)
    numax = snr.estimate_numax(numaxs=daynumaxs)
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))
    assert(numax.unit == u.microhertz)

def test_plot_numax_diagnostics():
    """Test if we can estimate numax using the diagnostics function, and that
    it returns a correct metric when requested
    """
    f, p, true_numax, _ = generate_test_spectrum()
    std = 0.25*true_numax/2.335  # The standard deviation of the mode envelope
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    numaxs = np.linspace(true_numax-2*std, true_numax+2*std, 500)

    numax, _ = snr.plot_numax_diagnostics()
    #Note: checks on the `numaxs` kwarg in `estimate_numax_kwargs` also apply
    #to this function, no need to check them twice.

    #Assert recovers numax within 10%
    assert(np.isclose(true_numax, numax.value, atol=.1*true_numax))
    #Assert numax has unit equal to input frequency unit
    assert(numax.unit == u.microhertz)

    # Sanity check that plotting works under all conditions
    numax, ax = snr.plot_numax_diagnostics()
    numax, ax = snr.plot_numax_diagnostics(numaxs=numaxs)
    daynumaxs = u.Quantity(numaxs*u.microhertz, 1/u.day)
    numax, ax = snr.plot_numax_diagnostics(numaxs=daynumaxs)

    #Check metric of appropriate length is returned
    _, _, metric = snr.plot_numax_diagnostics(numaxs=numaxs,return_metric=True)
    assert(len(metric) == len(numaxs))


def test_estimate_dnu_basics():
    """Test if we can estimate a dnu
    """
    f, p, _, true_dnu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    dnu = snr.estimate_dnu()

    #Assert recovers dnu within 25%
    assert(np.isclose(true_dnu, dnu.value, atol=.25*true_dnu))
    #Assert dnu has unit equal to input frequency unit
    assert(dnu.unit == u.microhertz)
    #Assert dnu estimator works when input frequency is not in microhertz
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    daysnr = SNRPeriodogram(fday, u.Quantity(p, None))
    dnu = daysnr.estimate_dnu()
    dnuday = u.Quantity(true_dnu*u.microhertz, 1/u.day)
    assert(np.isclose(dnuday.value, dnu.value, atol=.25*dnuday.value))

def test_estimate_dnu_kwargs():
    """Test if we can estimate a dnu using its various keyword arguments
    """
    f, p, _, true_dnu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))

    # Assert custom numax works
    numax = snr.estimate_numax()
    dnu = snr.estimate_dnu(numax)
    assert(np.isclose(dnu.value, true_dnu, atol=.25*true_dnu))

    # Assert you can't pass custom numax outside of appropriate range
    with pytest.raises(ValueError) as err:
        dnu = snr.estimate_dnu(numax= -5.)
    with pytest.raises(ValueError) as err:
        dnu = snr.estimate_dnu(numax=5000)

    # Assert it doesn't matter what units of frequency numax is passed in as
    daynumax = u.Quantity(numax.value*u.microhertz, 1/u.day)
    dnu = snr.estimate_dnu(numax=daynumax)
    assert(np.isclose(dnu.value, true_dnu, atol=.25*true_dnu))
    assert(dnu.unit == u.microhertz)

def test_plot_dnu_diagnostics():
    """Test if we can estimate numax using the diagnostics function, and that
    it returns a correct metric when requested
    """
    f, p, _, true_dnu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))

    # Assert custom numax works
    dnu, _ = snr.plot_dnu_diagnostics()
    assert(np.isclose(dnu.value, true_dnu, atol=.25*true_dnu))
    assert(dnu.unit == u.microhertz)

    #Note: checks on the `numax` kwarg in `estimate_dnu_kwargs` also apply
    #to this function, no need to check them twice.

    # Sanity check that plotting works under all conditions
    dnu, ax = snr.plot_dnu_diagnostics()
    dnu, ax = snr.plot_dnu_diagnostics(numax=numax)
    dnu, ax = snr.plot_dnu_diagnostics(numax=daynumax)

    #Check it plots when frequency is in days
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    daysnr = SNRPeriodogram(fday, u.Quantity(p, None))
    dnu, ax = daysnr.plot_dnu_diagnostics(numax=daynumax)

    #Check metric of appropriate length is returned
    _, _, metric = snr.plot_dnu_diagnostics(numax=numax, return_metric=True)
    assert(len(metric) == len(snr._autocorrelate(numax.value)))

def test_plot_echelle():
    f, p, numax, dnu = generate_test_spectrum()
    numax *= u.microhertz
    dnu *= u.microhertz

    pg = Periodogram(f*u.microhertz, u.Quantity(p, None))

    #Assert basic echelle works
    pg.plot_echelle(dnu)
    pg.plot_echelle(u.Quantity(dnu, 1/u.day))

    #Assert echelle works with numax
    pg.plot_echelle(dnu, numax)
    pg.plot_echelle(dnu, u.Quantity(numax, 1/u.day))

    #Assert echelle works with minimum limit
    pg.plot_echelle(dnu, minimum_frequency = numax)
    pg.plot_echelle(dnu, maximum_frequency = numax)
    pg.plot_echelle(dnu, minimum_frequency = u.Quantity(numax, 1/u.day))
    pg.plot_echelle(dnu, maximum_frequency = u.Quantity(numax, 1/u.day))
    pg.plot_echelle(dnu, minimum_frequency = u.Quantity(numax-dnu, 1/u.day),
                        maximum_frequency = numax+dnu)

    #Assert raises error if numax or either of the limits are too high
    with pytest.raises(ValueError) as err:
        pg.plot_echelle(dnu, minimum_frequency = f[-1]+10)
    with pytest.raises(ValueError) as err:
        pg.plot_echelle(dnu, maximum_frequency = f[-1]+10)
    with pytest.raises(ValueError) as err:
        pg.plot_echelle(dnu, numax = f[-1]+10)

    #Assert can pass colourmap
    pg.plot_echelle(dnu, cmap='viridis')

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
    assert mask.sum() > (~mask).sum()

    assert isinstance(p.period_at_max_power, u.Quantity)
    assert isinstance(p.duration_at_max_power, u.Quantity)
    assert isinstance(p.transit_time_at_max_power, float)
    assert isinstance(p.depth_at_max_power, float)


@pytest.mark.skipif(bad_optional_imports, reason="requires astropy.stats.bls")
def test_bls_period_recovery():
    """Can BLS Periodogram recover the period of a synthetic light curve?"""
    # Planet parameters
    period = 2.0
    transit_time = 0.5
    duration = 0.1
    depth = 0.2
    flux_err = 0.01

    # Create the synthetic light curve
    time = np.arange(0, 100, 0.1)
    flux = np.ones_like(time)
    transit_mask = np.abs((time-transit_time+0.5*period) % period-0.5*period) < 0.5*duration
    flux[transit_mask] = 1.0 - depth
    flux += flux_err * np.random.randn(len(time))
    synthetic_lc = LightCurve(time, flux)

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
    assert("parameter must be one of 'boxkernel' or 'logmedian'" in err.value.args[0])


@pytest.mark.skipif(bad_optional_imports, reason="requires astropy.stats.bls")
def test_bls_period():
    """Regression test for #514."""
    lc = LightCurve(time=[1, 2, 3], flux=[4, 5, 6])
    period = [1, 2, 3, 4, 5]
    pg = lc.to_periodogram(method="bls", period=period)
    assert_array_equal(pg.period.value, period)
    with pytest.raises(ValueError) as err:  # NaNs should raise a nice error message
        lc.to_periodogram(method="bls", period=[1, 2, 3, np.nan, 4])
    assert("period" in err.value.args[0])
