# Synthetic test data

These synthetic data were generated from the [scope package](https://github.com/nksaunders/scope), to simulate Target Pixel Files (TPFs) with known levels of spacecraft-induced roll motion.  

**synthetic-k2-flat.targ.fits.gz**  a simulated flat (featureless) lightcurve.

**synthetic-k2-planet.targ.fits.gz**  a simulated planet host star with the following properties:
-  Transit period 5.0 days
-  Rp/R_star 0.044
-  Planet semi-major axis in units of R_star 12.3108
-  Starry limb darkening coefficients (0.40, 0.26)

**synthetic-k2-sinusoid.targ.fits.gz** a simulated sine wave with:
-  Rotation period 5.0 days
-  Sine fractional amplitude 0.001 (i.e. 0.1% or 1000 ppm), so peak-to-valley 0.002.

The level of roll motion was mimicked from the K2 source EPIC 205998445, for a hypothetical 12th magnitude star.  1000 cadences were simulated.

The noise-free signals are housed in the HDU extension 3 "SIMULATED_SIGNAL", with a single data vector "NOISELESS_INPUT".

The *SIMULATED_SIGNAL* HDU header has additional header cards listing the values of the planet and sine wave parameters.


## Python code to generate this synthetic data with `scope`

You will need these dependencies:

```python
import scope
import lightkurve as lk
import numpy as np
import starry
from astropy.io import fits
import copy

target = scope.generate_target(ncadences=500)

```

### Sine wave signal

```Python
target_sine = copy.copy(target)

rot_period = 5.0 # days
lc_amp = 0.001 # percent amplitude
injected_sine_lc = 1.0+ lc_amp * np.sin(2.0*np.pi*target.t/rot_period)

target_sine.add_variability(custom_variability=injected_sine_lc)

sine_tpf = target_sine.to_lightkurve_tpf(target_id="simulated_sine_P5days_Amp0p001")

genuine_tpf = lk.search_targetpixelfile(205998445, mission='K2').download()
col = genuine_tpf.get_keyword('1CRV5P', hdu=1, default=0)
row = genuine_tpf.get_keyword('2CRV5P', hdu=1, default=0)

sine_tpf.hdu[1].header.set('1CRV5P', value=col)
sine_tpf.hdu[1].header.set('2CRV5P', value=row)

extra_hdu = fits.BinTableHDU.from_columns([fits.Column(name='NOISELESS_INPUT', format='E',
                                 array=injected_sine_lc)])

extra_hdu.name = 'SIMULATED_SIGNAL'

sine_tpf.hdu.append(extra_hdu)

sine_tpf.hdu[3].header.set('PERIOD', value=rot_period, comment='Period of noiseless input sine wave')
sine_tpf.hdu[3].header.set('SINE_AMP', value=lc_amp, comment='Amplitude of noiseless input sine wave')

sine_tpf.to_fits(output_fn='synthetic-k2-sinusoid.targ.fits', overwrite=True)

```


### Transiting exoplanet signal

```python
star = starry.kepler.Primary()

star[1] = 0.40
star[2] = 0.26

planet = starry.kepler.Secondary(lmax=5)

planet.tref = 2144.3

planet.r = 0.044 # Rp/R_star, reverse-engineering a nearly 2000 ppm transit without limb darkening
planet.porb = 5.0 # 5 day period
planet.a = 12.3108 # Semi-major axis, assuming 1 solar mass star

rprs = np.sqrt(2000/1.0e6)

system = starry.kepler.System(star, planet)

time = target.t
system.compute(time)

target_planet = copy.copy(target)

target_planet.add_variability(custom_variability=system.lightcurve)

plan_tpf = target_planet.to_lightkurve_tpf(target_id="simulated_planet_P5days")
plan_tpf.hdu[1].header.set('1CRV5P', value=col)
plan_tpf.hdu[1].header.set('2CRV5P', value=row)

extra_hdu = fits.BinTableHDU.from_columns([fits.Column(name='NOISELESS_INPUT', format='E',
                                 array=system.lightcurve)])
extra_hdu.name = 'SIMULATED_SIGNAL'
plan_tpf.hdu.append(extra_hdu)
plan_tpf.hdu[3].header.set('PERIOD', value=5.0, comment='Period of noiseless input transit')
plan_tpf.hdu[3].header.set('RPRS', value=0.044, comment='Rp/Rstar ')
plan_tpf.hdu[3].header.set('PLANA', value=12.3108, comment='Planet semi-major axis in R star')
plan_tpf.hdu[3].header.set('STARLD1', value=0.40, comment='Starry limb darkening 1')
plan_tpf.hdu[3].header.set('STARLD2', value=0.26, comment='Starry limb darkening 2')
plan_tpf.hdu[3].header.set('PLANTREF', value=2144.3, comment='Planet transit reference time')

plan_tpf.to_fits(output_fn='synthetic-k2-planet.targ.fits', overwrite=True)
```


### Default, flat lightcurve

```Python
flat_tpf = target.to_lightkurve_tpf(target_id="simulated_flat_star")
flat_tpf.hdu[1].header.set('1CRV5P', value=col)
flat_tpf.hdu[1].header.set('2CRV5P', value=row)

extra_hdu = fits.BinTableHDU.from_columns([fits.Column(name='NOISELESS_INPUT', format='E',
                                 array=np.ones(500))])
extra_hdu.name = 'SIMULATED_SIGNAL'
flat_tpf.hdu.append(extra_hdu)

flat_tpf.to_fits(output_fn='synthetic-k2-flat.targ.fits', overwrite=True)
```
