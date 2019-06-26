Synthetic test data
---

These synthetic data were generated from the [scope package](https://github.com/nksaunders/scope), to simulate Target Pixel Files (TPFs) with known levels of spacecraft-induced roll motion.  

**synthetic-k2-flat.targ.fits.gz**  a simulated flat (featureless) lightcurve.

**synthetic-k2-planet.targ.fits.gz**  a simulated planet host star with the following properties:
- Transit period 5.0 days
- Rp/R_star 0.044
- Planet semi-major axis in units of R_star 12.3108
- Starry limb darkening coefficients (0.40, 0.26)

**synthetic-k2-sinusoid.targ.fits.gz** a simulated sine wave with:
- Rotation period 5.0 days
- Sine fractional amplitude 0.001 (i.e. 0.1% or 1000 ppm), so peak-to-valley 0.002.


The level of roll motion was mimicked from the K2 source EPIC 205998445, for a hypothetical 12th magnitude star.
