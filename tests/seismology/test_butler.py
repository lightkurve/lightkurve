import pytest
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import unit_impulse as deltafn

from ...search import search_lightcurve
from ...periodogram import Periodogram
from ...periodogram import SNRPeriodogram


@pytest.mark.remote_data
def test_asteroseismology():
    datalist = search_lightcurve('KIC11615890')
    data = datalist.download_all()
    lc = data[0].normalize().flatten()
    for nlc in data[0:5]:
        lc = lc.append(nlc.normalize().flatten())
    lc = lc.remove_nans()
    pg = lc.to_periodogram(normalization='psd')
    snr = pg.flatten()
    snr.to_seismology().estimate_numax()


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

    deltanu_true = 0.294 * nmx ** 0.772
    modelocs = np.arange(lo, hi, deltanu_true/2, dtype=int)

    for modeloc in modelocs:
        m += deltafn(len(f), modeloc)
    p *= m
    p += 1
    return f, p, nmx, deltanu_true


def test_estimate_numax_basics():
    """Test if we can estimate a numax."""
    f, p, true_numax, _ = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    numax = snr.to_seismology().estimate_numax()

    #Assert recovers numax within 10%
    assert(np.isclose(true_numax, numax.value, atol=.1*true_numax))
    #Assert numax has unit equal to input frequency unit
    assert(numax.unit == u.microhertz)

    # Assert you can recover numax with a chopped periodogram
    rsnr = snr[(snr.frequency.value>1600) & (snr.frequency.value<3200)]
    numax = rsnr.to_seismology().estimate_numax()
    assert(np.isclose(true_numax, numax.value, atol=.1*true_numax))

    # Assert numax estimator works when input frequency is not in microhertz
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    snr = SNRPeriodogram(fday, u.Quantity(p, None))
    numax = snr.to_seismology().estimate_numax()
    nmxday = u.Quantity(true_numax*u.microhertz, 1/u.day)
    assert(np.isclose(nmxday, numax, atol=.1*nmxday))

    # Assert numax estimator fails when frequqencies are not uniform
    f, p, true_numax, _ = generate_test_spectrum()
    f += np.random.uniform(size=len(f))
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))

    with pytest.raises(ValueError) as exc:
        numax = snr.to_seismology().estimate_numax()
    assert "uniformly spaced" in str(exc.value)

def test_estimate_numax_kwargs():
    """Test if we can estimate a numax using its various keyword arguments."""
    f, p, true_numax, _ = generate_test_spectrum()
    std = 0.25*true_numax/2.335  # The standard deviation of the mode envelope
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    butler = snr.to_seismology()
    numaxs = np.linspace(true_numax-2*std, true_numax+2*std, 500)
    numax = butler.estimate_numax(numaxs=numaxs)

    # Assert we can recover numax using a custom numax
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))

    # Assert we can't pass custom numaxs outside a functional range
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(numaxs=np.linspace(-5, 5.))
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(numaxs=np.linspace(1., 5000.))

    # Assert we can pass a custom window in microhertz or days
    numax = butler.estimate_numax(window_width=200.)
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))
    numax = butler.estimate_numax(window_width=u.Quantity(200., u.microhertz).to(1/u.day))
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))

    # Assert we can't pass in window_widths outside functional range
    # Assert we can't pass custom numaxs outside a functional range
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(window_width=-5)
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(window_width=1e6)
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(window_width=0.001)

    # Assert we can pass a custom spacing in microhertz or days
    numax = butler.estimate_numax(spacing=15.)
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))
    numax = butler.estimate_numax(spacing=u.Quantity(15., u.microhertz).to(1/u.day))
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))

    # Assert we can't pass in spacing outside functional range
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(spacing=-5)
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(spacing=1e6)
    with pytest.raises(ValueError):
        numax = butler.estimate_numax(spacing=0.001)

    # Assert it doesn't matter what units of frqeuency numaxs are passed in as
    # Assert the output is still in the same units as the object frequencies
    daynumaxs = u.Quantity(numaxs*u.microhertz, 1/u.day)
    numax = butler.estimate_numax(numaxs=daynumaxs)
    assert(np.isclose(numax.value, true_numax, atol=.1*true_numax))
    assert(numax.unit == u.microhertz)


def test_plot_numax_diagnostics():
    """Test if we can estimate numax using the diagnostics function, and that
    it returns a correct metric when requested
    """
    f, p, true_numax, _ = generate_test_spectrum()
    std = 0.25*true_numax/2.335  # The standard deviation of the mode envelope
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    butler = snr.to_seismology()
    numaxs = np.linspace(true_numax-2*std, true_numax+2*std, 500)
    butler.estimate_numax(numaxs=numaxs, window_width=250., spacing=10.)
    butler.diagnose_numax()
    # Note: checks on the `numaxs` kwarg in `estimate_numax_kwargs` also apply
    # to this function, no need to check them twice.

    # Assert recovers numax within 10%
    assert(np.isclose(true_numax, butler.numax.value, atol=.1*true_numax))
    # Assert numax has unit equal to input frequency unit
    assert(butler.numax.unit == u.microhertz)

    # Sanity check that plotting works under all conditions
    numax = butler.estimate_numax()
    butler.diagnose_numax(numax)
    numax = butler.estimate_numax(numaxs=numaxs)
    butler.diagnose_numax(numax)
    daynumaxs = u.Quantity(numaxs*u.microhertz, 1/u.day)
    numax = butler.estimate_numax(numaxs=daynumaxs)
    butler.diagnose_numax(numax)
    numax = butler.estimate_numax(window_width=100.)
    butler.diagnose_numax(numax)

    # Check plotting works when periodogram is sliced
    rsnr = snr[(snr.frequency.value>1600)&(snr.frequency.value<3200)]
    butler = rsnr.to_seismology()
    butler.estimate_numax()
    butler.diagnose_numax()

    # Check metric of appropriate length is returned
    numax = butler.estimate_numax(numaxs=numaxs)
    assert(len(numax.diagnostics['metric']) == len(numaxs))


def test_estimate_deltanu_basics():
    """Test if we can estimate a deltanu
    """
    f, p, _, true_deltanu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    butler = snr.to_seismology()
    butler.estimate_numax()
    deltanu = butler.estimate_deltanu()

    # Assert recovers deltanu within 25%
    assert(np.isclose(true_deltanu, deltanu.value, atol=.25*true_deltanu))
    # Assert deltanu has unit equal to input frequency unit
    assert(deltanu.unit == u.microhertz)

    # Assert you can recover numax with a sliced periodogram
    rsnr = snr[(snr.frequency.value>1600) & (snr.frequency.value<3200)]
    butler = rsnr.to_seismology()
    butler.estimate_numax()
    numax = butler.estimate_deltanu()
    assert(np.isclose(true_deltanu, deltanu.value, atol=.25*true_deltanu))

    # Assert deltanu estimator works when input frequency is not in microhertz
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    daysnr = SNRPeriodogram(fday, u.Quantity(p, None))
    butler = daysnr.to_seismology()
    butler.estimate_numax()
    deltanu = butler.estimate_deltanu()
    deltanuday = u.Quantity(true_deltanu*u.microhertz, 1/u.day)
    assert(np.isclose(deltanuday.value, deltanu.value, atol=.25*deltanuday.value))

    # Assert deltanu estimator fails when frequqencies are not uniform
    f, p, true_numax, _ = generate_test_spectrum()
    f += np.random.uniform(size=len(f))
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))

    with pytest.raises(ValueError) as exc:
        deltanu = snr.to_seismology().estimate_deltanu(numax=100)
    assert "uniformly spaced" in str(exc.value)

def test_estimate_deltanu_kwargs():
    """Test if we can estimate a deltanu using its various keyword arguments
    """
    f, p, _, true_deltanu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    butler = snr.to_seismology()

    # Assert custom numax works
    numax = butler.estimate_numax()
    deltanu = butler.estimate_deltanu(numax=numax)
    assert(np.isclose(deltanu.value, true_deltanu, atol=.25*true_deltanu))

    # Assert you can't pass custom numax outside of appropriate range
    with pytest.raises(ValueError):
        deltanu = butler.estimate_deltanu(numax= -5.)
    with pytest.raises(ValueError):
        deltanu = butler.estimate_deltanu(numax=5000)

    # Assert it doesn't matter what units of frequency numax is passed in as
    daynumax = u.Quantity(numax.value*u.microhertz, 1/u.day)
    deltanu = butler.estimate_deltanu(numax=daynumax)
    assert(np.isclose(deltanu.value, true_deltanu, atol=.25*true_deltanu))
    assert(deltanu.unit == u.microhertz)


def test_plot_deltanu_diagnostics():
    """Test if we can estimate numax using the diagnostics function, and that
    it returns a correct metric when requested
    """
    f, p, _, true_deltanu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    butler = snr.to_seismology()

    butler.estimate_numax()
    deltanu = butler.estimate_deltanu()
    ax = butler.diagnose_deltanu()
    assert(np.isclose(deltanu.value, true_deltanu, atol=.25*true_deltanu))
    assert(deltanu.unit == u.microhertz)
    plt.close('all')

    #Note: checks on the `numax` kwarg in `estimate_deltanu_kwargs` also apply
    #to this function, no need to check them twice.

    # Sanity check that plotting works under all conditions
    numax = butler.estimate_numax()
    butler.diagnose_deltanu()
    deltanu = butler.estimate_deltanu(numax=numax)
    butler.diagnose_deltanu(deltanu)
    daynumax = u.Quantity(numax.value*u.microhertz, 1/u.day)
    deltanu = butler.estimate_deltanu(numax=daynumax)
    butler.diagnose_deltanu(deltanu)
    plt.close('all')

    # Check plotting works when periodogram is sliced
    rsnr = snr[(snr.frequency.value>1600)&(snr.frequency.value<3200)]
    butler = rsnr.to_seismology()
    butler.estimate_numax()
    butler.estimate_deltanu()
    ax = butler.diagnose_deltanu()
    plt.close('all')

    # Check it plots when frequency is in days
    fday = u.Quantity(f*u.microhertz, 1/u.day)
    daysnr = SNRPeriodogram(fday, u.Quantity(p, None))
    butler = daysnr.to_seismology()
    butler.estimate_deltanu(numax=daynumax)
    butler.diagnose_deltanu()
    plt.close('all')


def test_stellar_estimator_calls():
    f, p, _, true_deltanu = generate_test_spectrum()
    snr = SNRPeriodogram(f*u.microhertz, u.Quantity(p, None))
    snr.meta = {'TEFF': 3000}

    butler = snr.to_seismology()
    butler.estimate_numax()
    deltanu = butler.estimate_deltanu()

    # Calling teff from meta
    mass = butler.estimate_mass()
    rad = butler.estimate_radius()
    log = butler.estimate_logg()

    # Custom teff
    mass = butler.estimate_mass(3100)
    rad = butler.estimate_radius(3100)
    log = butler.estimate_logg(3100)

    # Raise error if no teff available
    butler.periodogram.meta['TEFF'] = None
    with pytest.raises(ValueError):
        mass = butler.estimate_mass()
    with pytest.raises(ValueError):
        rad = butler.estimate_radius()
    with pytest.raises(ValueError):
        log = butler.estimate_logg()

def test_plot_echelle():
    f, p, numax, deltanu = generate_test_spectrum()
    numax *= u.microhertz
    deltanu *= u.microhertz

    pg = Periodogram(f*u.microhertz, u.Quantity(p, None))
    butler = pg.to_seismology()

    # Assert basic echelle works
    butler.plot_echelle(deltanu=deltanu, numax=numax)
    plt.close('all')
    butler.plot_echelle(u.Quantity(deltanu, 1/u.day), numax)
    plt.close('all')

    # Assert accepts dimensionless input
    butler.plot_echelle(deltanu=deltanu.value*1.001, numax=numax)
    plt.close('all')
    butler.plot_echelle(deltanu=deltanu, numax=numax.value/1.001)
    plt.close('all')

    # Assert echelle works with numax
    butler.plot_echelle(deltanu, numax)
    plt.close('all')
    butler.plot_echelle(deltanu, u.Quantity(numax, 1/u.day))
    plt.close('all')

    # Assert echelle works with minimum limit
    butler.plot_echelle(deltanu, numax, minimum_frequency=numax)
    plt.close('all')
    butler.plot_echelle(deltanu, numax, maximum_frequency=numax)
    plt.close('all')
    butler.plot_echelle(deltanu, numax, minimum_frequency=u.Quantity(numax, 1/u.day))
    plt.close('all')
    butler.plot_echelle(deltanu, numax, maximum_frequency=u.Quantity(numax, 1/u.day))
    plt.close('all')
    butler.plot_echelle(deltanu, numax, minimum_frequency=u.Quantity(numax-deltanu, 1/u.day),
                        maximum_frequency=numax+deltanu)
    plt.close('all')

    # Assert raises error if numax or either of the limits are too high
    with pytest.raises(ValueError):
        butler.plot_echelle(deltanu, numax, minimum_frequency=f[-1]+10)
    plt.close('all')
    with pytest.raises(ValueError):
        butler.plot_echelle(deltanu, numax, maximum_frequency=f[-1]+10)
    plt.close('all')
    with pytest.raises(ValueError):
        butler.plot_echelle(deltanu, numax=f[-1]+10)
    plt.close('all')

    # Assert can pass colormap
    butler.plot_echelle(deltanu, numax, cmap='viridis')
    plt.close('all')
