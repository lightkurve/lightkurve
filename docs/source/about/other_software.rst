.. _other_software:

==============
Other software
==============

Lightkurve provides general purpose tools for interacting with astronomical lightcurve data.
Many other tools have been developed to solve related scientific and data analysis problems.
On this page we list community-produced software that may complement lightkurve.

If your software is not listed, please `open a Pull Request <https://github.com/KeplerGO/lightkurve/blob/master/docs/source/other_software.rst>`_ to add it, we aim to be inclusive of all Kepler- and TESS-related tools!


Detrending & Analysis
~~~~~~~~~~~~~~~~~~~~~~

- `VARTOOLS <https://www.astro.princeton.edu/~jhartman/vartools.html>`_ : a command line utility to analyze light curves from Hartman and Bakos (2016).
- `PyKE <http://github.com/KeplerGO/PyKE>`_ : Kepler, K2 & TESS Data Analysis Tools (the precursor to Lightkurve)
- `everest <http://github.com/rodluger/everest>`_ : De-trending of K2 Light curves
- `k2sc <http://github.com/OxES/k2sc>`_ : K2 systematics correction using Gaussian processes
- `nutella <http://github.com/benmontet/nutella>`_ : Great (point) spreads for beautiful Kepler/K2 inference
- `skope <http://github.com/nksaunders/skope>`_ : Synthetic K2 Objects for PLD Experimentation
- `k2phot <http://github.com/petigura/k2phot>`_ : public k2phot code from Erik Petigura
- `K2-CPM <http://github.com/jvc2688/K2-CPM>`_ : K2 Causal Pixel Model
- `halophot <https://github.com/hvidy/halophot/>`_ : K2 Halo Photometry for very bright stars
- `cave <http://github.com/nksaunders/cave>`_ : Crowded Aperture Variability Extraction
- `celerite-asteroseis <http://github.com/skgrunblatt/celerite-asteroseis>`_ : Transit fitting and basic time-domain asteroseismology using celerite and ktransit
- `k2photometry <http://github.com/vincentvaneylen/k2photometry>`_ : Read, reduce and detrend K2 photometry and search for transiting planets
- `keplersmear <http://github.com/benjaminpope/keplersmear>`_ : Make light curves from Kepler and K2 collateral data
- `OxKeplerSC <http://github.com/OxES/OxKeplerSC>`_ : Kepler jump and systematics correction using Variational Bayes and shrinkage priors.
- `K2Pipeline <http://github.com/FGCUStellarResearch/K2Pipeline>`_ : Data reduction and detrending pipeline for K2 data in Matlab
- `PySysRem <http://github.com/stephtdouglas/PySysRem>`_ : A Python implementation of the SysRem algorithm from Tamuz, Mazeh, and Zucker (2004)
- `CBVshrink <https://github.com/saigrain/CBVshrink>`_ : Kepler systematics correction using co-trending basis vectors (CBV), Variational Bayes and shrinkage priors


Full Frame Images
~~~~~~~~~~~~~~~~~~

- `f3 <http://github.com/benmontet/f3>`_ : Full Frame Fotometry from the Kepler Full Frame Images
- `FFIorBUST <http://github.com/jradavenport/FFIorBUST>`_ : Make really bad light curves from the Kepler Full Frame Images
- `kepcal <http://github.com/dfm/kepcal>`_ : Self calibration using the Kepler FFIs


Data access
~~~~~~~~~~~~

- `kplr <http://github.com/dfm/kplr>`_ : Tools for working with Kepler data using Python
- `kepFGS <http://github.com/christinahedges/kepFGS>`_ : Tools to use the Kepler and K2 Fine Guidance Sensor data.
- `k2plr <http://github.com/rodluger/k2plr>`_ : Fork of dfm/kplr with added k2 functionality


Metadata
~~~~~~~~~

- `kadenza <http://github.com/KeplerGO/kadenza>`_ : Converts raw cadence target data from the Kepler space telescope into FITS files.
- `k2-quality-control <http://github.com/KeplerGO/k2-quality-control>`_ : Automated quality control of Kepler/K2 data products.
- `SuperstampFITS <http://github.com/amcody/SuperstampFITS>`_ : Create individual FITS files of K2 superstamp regions.
- `keputils <http://github.com/timothydmorton/keputils>`_ : Basic module for interaction with KOI and Kepler-stellar tables.


Planet Search/Characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `starry <https://github.com/rodluger/starry>`_ : Analytic occultation light curves for astronomy.
- `PyTransit <http://github.com/hpparvi/PyTransit>`_ : Fast and easy transit light curve modeling using Python and Fortran.
- `batman <http://github.com/lkreidberg/batman>`_ : Fast transit light curves models in Python.
- `robin <https://robin.readthedocs.io/en/latest/>`_ : Robust exoplanet radii from ingress/egress durations
- `ktransit <http://github.com/mrtommyb/ktransit>`_ : A simple exoplanet transit modeling tool in python
- `planetplanet <http://github.com/rodluger/planetplanet>`_ : A general photodynamical code for exoplanet light curves
- `ketu <http://github.com/dfm/ketu>`_ : Search for transiting planets in K2 data
- `ttvfast-python <http://github.com/mindriot101/ttvfast-python>`_ : Python interface to the TTVFast library
- `TTV2Fast2Furious <https://github.com/shadden/TTV2Fast2Furious>`_ : Construct and fit linear transit timing variation models
- `terra <http://github.com/petigura/terra>`_ : Transit Detection Code
- `pysyzygy <http://github.com/rodluger/pysyzygy>`_ : A fast and general planet transit (syzygy) code written in C and in Python
- `wellfit <https://github.com/christinahedges/wellfit>`_ : Turnkey transit modeling with starry and celerite
- `k2ps <http://github.com/hpparvi/k2ps>`_ : K2 Planet Search
- `lcps <http://github.com/matiscke/lcps>`_ : A tool for pre-selecting light curves with possible transit signatures


Population Statistics
~~~~~~~~~~~~~~~~~~~~~~

- `VESPA <http://github.com/timothydmorton/VESPA>`_ : Calculating false positive probabilities for transit signals
- `kepler-robovetter <http://github.com/nasa/kepler-robovetter>`_ : The Kepler Prime Robovetter
- `koi-fpp <http://github.com/timothydmorton/koi-fpp>`_ : False positive probabilities for all KOIs
- `KeplerPORTS <http://github.com/nasa/KeplerPORTS>`_ : The Kepler Pipeline
- `Kepler-FLTI <http://github.com/nasa/Kepler-FLTI>`_ : Kepler Prime Flux-Level Transit Injection
- `epos <https://github.com/GijsMulders/epos>`_ : Exoplanet Population Observation Simulator


Positional
~~~~~~~~~~~

- `K2fov <http://github.com/KeplerGO/K2fov>`_ : Check whether targets are in the field of view of NASA's K2 space telescope
- `K2ephem <http://github.com/KeplerGO/K2ephem>`_ : Check whether a Solar System body is (or was) observable by NASA's K2 mission.
- `k2-pix <http://github.com/stephtdouglas/k2-pix>`_ : Overlay a sky survey image on a K2 target pixel stamp
- `k2flix <http://github.com/barentsen/k2flix>`_ : Create quicklook movies from the pixel data observed by Kepler/K2/TESS
- `k2mosaic <http://github.com/barentsen/k2mosaic>`_ : Mosaic Target Pixel Files (TPFs) obtained by NASA's Kepler/K2 missions into images and movies.
- `gaia-kepler.fun <https://github.com/megbedell/gaia-kepler.fun>`_ : Gaia DR2 + Kepler/K2 cross-matches
- `tvguide <http://github.com/tessgi/tvguide>`_ : A tool for determining whether stars and galaxies are observable by TESS.


Science / Astrophysics
~~~~~~~~~~~~~~~~~~~~~~~

- `isochrones <http://github.com/timothydmorton/isochrones>`_ : Pythonic stellar model grid access; easy MCMC fitting of stellar properties
- `ldtk <http://github.com/hpparvi/ldtk>`_ : Python toolkit for calculating stellar limb darkening profiles
- `isoclassify <http://github.com/danxhuber/isoclassify>`_ : Perform stellar classifications using isochrone grids
- `appaloosa <http://github.com/jradavenport/appaloosa>`_ : Python-based flare finding code for Kepler light curves.
- `pymacula <http://github.com/timothydmorton/pymacula>`_ : Python wrapper for Macula analytic starspot code
- `MulensModel <http://github.com/rpoleski/MulensModel>`_ : Microlensing Modelling package
- `animate_spots <http://github.com/stephtdouglas/animate_spots>`_ : Make frames for animated gifs/movies showing a rotating spotted star
- `asteriks <https://github.com/christinahedges/asteriks>`_ : Generate light curves of solar system objects from K2 data
- `decatur <http://github.com/jadilia/decatur>`_ : Tidal Synchronization of Kepler Eclipsing Binaries
- `asteroseismology <https://github.com/earlbellinger/asteroseismology>`_ : Forward and inverse problems in asteroseismology


Astronomical Spectroscopy / Radial Velocities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `exoplanet <https://github.com/dfm/exoplanet>`_ : Fast and scalable MCMC for all your exoplanet needs
- `exonailer <https://github.com/nespinoza/exonailer>`_ : Tools for fitting transiting exoplanet lightcurves and radial velocities
- `pyaneti <https://github.com/oscaribv/pyaneti>`_ : A multi-planet Radial Velocity and Transit fit software
- `radvel <http://github.com/California-Planet-Search/radvel>`_ : General Toolkit for Modeling Radial Velocity Data
- `PyORBIT <https://github.com/LucaMalavolta/PyORBIT>`_ : Simultaneously characterize the orbits of exoplanets and the noise induced by stellar activity.
- `wobble <https://github.com/megbedell/wobble>`_ : Precise data-driven RV fitting with treatment for telluric contamination
- `Starfish <https://github.com/iancze/Starfish>`_ : Tools for Flexible Spectroscopic Inference
- `PSOAP <https://github.com/iancze/PSOAP>`_ : Tools for data-driven spectra models with Gaussian processes
- `specmatch-emp <https://github.com/samuelyeewl/specmatch-emp>`_ : Spectral matching with empirical templates
- `specmatch-syn <https://github.com/petigura/specmatch-syn>`_ : Spectral matching with synthetic templates


Other
~~~~~~

- `PandExo <http://github.com/natashabatalha/PandExo>`_ : A Community Tool for Transiting Exoplanet Science with the JWST & HST
- `kepler_orrery <http://github.com/ethankruse/kepler_orrery>`_ : Make a Kepler orrery gif or movie of all the Kepler multi-planet systems
- `orbitize <https://github.com/sblunt/orbitize>`_ : Orbit-fitting for directly imaged objects
- `tango <https://github.com/oscaribv/tango>`_ : Animate exoplanet transit orbits on a stellar disk
- `koi3278 <http://github.com/ethankruse/koi3278>`_ : Analysis files for the KOI-3278 system
- `trappist1 <http://github.com/rodluger/trappist1>`_ : TRAPPIST-1 photometry with K2
