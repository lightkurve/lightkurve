.. _other_software:

==============
Other software
==============

Lightkurve provides general purpose tools for interacting with astronomical lightcurve data.
Many other tools have been developed to solve related scientific and data analysis problems.
On this page we list community-produced software that may complement lightkurve.

If your software is not listed, please `open a Pull Request <https://github.com/KeplerGO/lightkurve/blob/master/docs/source/other_software.rst>`_ to add it!


Detrending & Analysis
~~~~~~~~~~~~~~~~~~~~~~
- `PyKE <http://github.com/KeplerGO/PyKE>`_ : Kepler, K2 & TESS Data Analysis Tools
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

Planet Search
~~~~~~~~~~~~~~

- `PyTransit <http://github.com/hpparvi/PyTransit>`_ : Fast and easy transit light curve modeling using Python and Fortran.
- `batman <http://github.com/lkreidberg/batman>`_ : Fast transit light curves models in Python.
- `ktransit <http://github.com/mrtommyb/ktransit>`_ : A simple exoplanet transit modeling tool in python
- `planetplanet <http://github.com/rodluger/planetplanet>`_ : A general photodynamical code for exoplanet light curves
- `ketu <http://github.com/dfm/ketu>`_ : I can haz planetz?
- `ttvfast-python <http://github.com/mindriot101/ttvfast-python>`_ : Python interface to the TTVFast library
- `terra <http://github.com/petigura/terra>`_ : Transit Detection Code
- `pysyzygy <http://github.com/rodluger/pysyzygy>`_ : A fast and general planet transit (syzygy) code written in C and in Python
- `k2ps <http://github.com/hpparvi/k2ps>`_ : K2 Planet Search
- `lcps <http://github.com/matiscke/lcps>`_ : A tool for pre-selecting light curves with possible transit signatures


Population Statistics
~~~~~~~~~~~~~~~~~~~~~~

- `VESPA <http://github.com/timothydmorton/VESPA>`_ : Calculating false positive probabilities for transit signals
- `kepler-robovetter <http://github.com/nasa/kepler-robovetter>`_ : The Kepler Prime Robovetter
- `koi-fpp <http://github.com/timothydmorton/koi-fpp>`_ : False positive probabilities for all KOIs
- `KeplerPORTS <http://github.com/nasa/KeplerPORTS>`_ : The Kepler Pipeline
- `Kepler-FLTI <http://github.com/nasa/Kepler-FLTI>`_ : Kepler Prime Flux-Level Transit Injection


Positional
~~~~~~~~~~~

- `K2fov <http://github.com/KeplerGO/K2fov>`_ : Check whether targets are in the field of view of NASA's K2 space telescope
- `K2ephem <http://github.com/KeplerGO/K2ephem>`_ : Check whether a Solar System body is (or was) observable by NASA's K2 mission.
- `k2-pix <http://github.com/stephtdouglas/k2-pix>`_ : Overlay a sky survey image on a K2 target pixel stamp
- `k2flix <http://github.com/barentsen/k2flix>`_ : Create quicklook movies from the pixel data observed by Kepler/K2/TESS
- `k2mosaic <http://github.com/barentsen/k2mosaic>`_ : Mosaic Target Pixel Files (TPFs) obtained by NASA's Kepler/K2 missions into images and movies.
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
- `decatur <http://github.com/jadilia/decatur>`_ : Tidal Synchronization of Kepler Eclipsing Binaries


Other
~~~~~~

- `PandExo <http://github.com/natashabatalha/PandExo>`_ : A Community Tool for Transiting Exoplanet Science with the JWST & HST
- `kepler_orrery <http://github.com/ethankruse/kepler_orrery>`_ : Make a Kepler orrery gif or movie of all the Kepler multi-planet systems
- `radvel <http://github.com/California-Planet-Search/radvel>`_ : General Toolkit for Modeling Radial Velocity Data
- `koi3278 <http://github.com/ethankruse/koi3278>`_ : Analysis files for the KOI-3278 system
- `trappist1 <http://github.com/rodluger/trappist1>`_ : TRAPPIST-1 photometry with K2
- `PyORBIT <https://github.com/LucaMalavolta/PyORBIT>`_: Simultaneously characterize the orbits of exoplanets and the noise induced by stellar activity.
