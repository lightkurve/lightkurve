2.0.0 (unreleased)
==================


Major changes
-------------

- Removed support for Python 2. [#733]

- ``LightCurve`` is now a sub-class of ``astropy.time.TimeSeries`` to enable it
  to have arbitrary columns and to enable closer integration with AstroPy. [#744]

- Updated the online docs with a new template, a dozen new tutorials, and a
  re-organized API reference guide. [#926]

Other changes
-------------

lightkurve.lightcurve
^^^^^^^^^^^^^^^^^^^^^

- Added a ``column`` parameter to ``LightCurve``'s ``plot()``, ``scatter()``,
  and ``errorbar()`` methods to enable any column to be plotted. [#765]

- Added the ``LightCurve.create_transit_mask(period, transit_time, duration)``
  method to conveniently mask planet or eclipsing binary transits. [#808]

- Added a ``column`` parameter to ``LightCurve.remove_nans()`` to enable
  cadences to be removed which contain NaN values in a specific column. [#828]

- ``interact_bls()``: added the support zoom by scrolling mouse wheel. [#854]

- ``interact_bls()``: modified so that it normalizes the lightcurve to match the
  generated transit model.  [#854]

- Fixed a bug in ``interact_bls()`` that caused the LightCurve panel improperly
  scaled. [#902]

- Added the ``LightCurve.search_neighbors()`` convenience method to search for
  light curves around an existing one. [#907]

lightkurve.targetpixelfile
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Added the ability for the user to specify how they want the flux calculated
  using ``extract_aperture_photometry``. The default is 'sum', but they
  there is now the option of using a ``flux_method`` keyword to specifiy if the
  user wants ``sum``, ``median``, or ``mean``. [#932]

- Added the ability to perform math with ``TargetPixelFile`` objects, e.g.,
  ``tpf = tpf - 100`` will subtract 100 from the ``tpf.flux`` values. [#665]

- Added the ``TargetPixelFile.plot_pixels()`` method to plot light curves
  and periodograms for each individual pixel in a TPF. [#771]

- Added the ``estimate_background`` method to ``TargetPixelFile`` which returns
  a 1D estimate of the residual background present in e.g. TESSCut data. [#746]

- Added a ``column`` parameter to ``TargetPixelFile.plot()`` to enable any
  column in a pixel file to be plotted (e.g. ``column="BKG_FLUX"``). [#738]

- Modified ``to_lightcurve()`` to default to using ``aperture_mask='threshold'``
  if the ``'pipeline'`` mask is missing or empty, e.g. for TESSCut files. [#833]

- Modified ``plot()`` to use a more clear hatched style when visualizing the
  aperture mask on top of pixel data. [#814]

- Modified ``_parse_aperture_mask()`` to ensure that masks composed of integer
  or floats are always converted to booleans. [#694]

- Fixed a bug in ``TargetPixelFile.__getitem__()`` which caused a substantial
  memory leak on indexing or slicing a tpf. [#829]

- Modified ``interact()`` to use ``max_cadences=200000`` by default to allow
  it to be used on fast-cadence TESS data. [#856]

- Modified `TargetPixelFactory` to support creating TESS Target Pixel Files
  and to enable it to populate all data columns. [#768, #857]

- Fixed a bug in ``TargetPixelFile.wcs`` which caused it to raise Error if
  the tpf does not contain expected WCS keywords in the header. [#892]

lightkurve.collections
^^^^^^^^^^^^^^^^^^^^^^

- Added the ability to filter a collection by `quarter`, `campaign` or `sector`. [#815]

lightkurve.search
^^^^^^^^^^^^^^^^^

- Added support for the new 20-second and 10-minute TESS cadence modes in the
  search functions by allowing the exact exposure time to be specified via the
  optional ``cadence`` argument.  In addition, the functions now also accept
  ``cadence='fast'`` (for 20s) and ``cadence='ffi'`` (for 10m or 30m). [#831]

- Modified the search functions to show the total exposure time of each data
  product (``t_exptime``) in the search results table. [#831]

- Added support for reading K2SFF and EVEREST community light curves via the
  ``LightCurve.read()`` and ``search_lightcurve()`` functions. [#739]

- Modified the search functions such that exact mission target identifiers,
  such as 'KIC 5112705' or 'TIC 261136679', only return products known under
  those names, unless a search radius is specified. [#796]

- Added support for searching and reading QLP and SPOC Full Frame Image (FFI)
  light curves available as High Level Science Products from MAST. [#913]

- Improved the performance of `download()` operations by checking if a file
  exists in local cache prior to contacting MAST. [#915]

- Added automated caching of the search operations. [#907]

- Modified the search operations to show all available data products at
  MAST by default, including community-contributed light curves. [#933]

lightkurve.correctors
^^^^^^^^^^^^^^^^^^^^^

- Added the ``CotrendingBasisVectors`` class to provide a convenient interface
  to work with TESS and Kepler basis vector data products. [#826]

- Changed the ``CBVCorrector`` class to perform the correction in a way that is
  more similar to the official Kepler/TESS pipeline. [#855]

- Added ``SparseDesignMatrix`` and modified ``RegressionCorrector`` to enable
  systematics removal methods to benefit from ``scipy.sparse`` speed-ups. [#732]

- Modified ``PLDCorrector`` to make use of the new ``RegressionCorrector``
  and ``DesignMatrix`` classes. [#746, #847]

- Fixed a bug in ``SFFCorrector`` which caused correction to fail if a light
  curve's ``centroid_col`` or ``centroid_row`` columns contained NaNs. [#827]

- Improved the ``Corrector`` abstract base class to better document the desired
  structure of its sub-classes. [#907]

- Added a ``metrics`` module with two functions to measure the degree of
  over- and under-fitting of a corrected light curve. [#907]

lightkurve.seismology
^^^^^^^^^^^^^^^^^^^^^

- Modified the ``estimate_radius``, ``estimate_mass``, and ``estimate_logg``
  methods to default to the ``teff`` value in the meta data. [#766]

- Added an error message to ``estimate_numax()`` or ``estimate_deltanu()`` if
  the underlying periodogram does not have uniformly-spaced frequencies. [#780]

lightkurve.periodogram
^^^^^^^^^^^^^^^^^^^^^^

- Modified ``create_transit_mask`` method to return ``True`` during transits and
  ``False`` elsewhere for consistent mask syntax. [#808]

- Modified ``BoxLeastSquaresPeriodogram`` to use ``duration=[0.05, 0.10, 0.15, 0.20, 0.25, 0.33]``
  by default, which yields more accurate results (albeit slower). [#859, #860]



1.11.3 (2020-10-06)
===================

- Fixed inline plots not appearing in Jupyter Notebooks and Google Colab. [#865]



1.11.2 (2020-08-28)
===================

- Fixed a warning being issued (``"LightCurveFile.header is deprecated"``)
  when downloading light curve files from MAST. [#819]



1.11.1 (2020-06-18)
===================

- Fixed a bug in ``TargetPixelFile.cutout()`` which prevented image edges from
  being included in cut-outs. [#749]

- Fixed a bug in ``tpf.interact()`` which caused the pixel selection to be off
  by half a pixel. The bug was introduced in v1.11.0. [#754]

- Fixed ``tpf.plot()`` and ``tpf.interact_sky()`` to reflect that Kepler and
  TESS pixel coordinates refer to pixel centers. [#755]

- Fixed broken links in tutorials. [#756]



1.11.0 (2020-05-20)
===================

- Deprecated the ``TargetPixelFile.header`` property and ``LightCurveFile.header()``
  method in favor of a consistent ``get_header()`` method. [#736]

- Fixed a bug in ``tpf.interact_sky()`` which caused star positions to be off
  by half a pixel. [#734]



1.10.0 (2020-05-14)
===================

- Added the ``query_solar_system_objects()`` method to search for solar system
  objects in ``TargetPixelFile`` and ``LightCurve`` objects. [#714]

- Added the ``extra_columns`` attribute to ``LightCurve`` objects. [#724]

- Fixed the URL to the Point Response Function (PRF) files in ``KeplerPRF``. [#727]

- Fixed a bug which caused searches to fail with Astroquery v0.4.1 and later. [#728]

- Fixed a bug in ``TargetPixelFile.interact_sky()`` which caused high proper
  motion stars to be shown at incorrect locations. [#730]



1.9.1 (2020-03-25)
==================

- Increased the speed of ``search_lightcurvefile()`` and
  ``search_targetpixelfile()`` by a factor ~10x. [#715]

- Fixed an issue which caused ``interact()`` and ``interact_bls()`` to be
  incompatible with Bokeh v2.0.0. [#716]

- Fixed a bug in `LightCurve.bin()` which caused the method to fail if the
  ``quality`` array has a floating point data type. [#705]



1.9.0 (2020-02-25)
==================

- Added an experimental ``TessPLDCorrector`` class designed to correct TESS FFI
  light curves by detrending against local pixel time series. [#687]

- Added a ``LightCurve.plot_river()`` method to plot river diagrams, which uses
  colors to visualize fluxes by period cycle (row) and phase (column). [#625]

- Added caching to `search_tesscut` to avoid requesting an identical cut out
  more than once. [#481]



1.8.0 (2020-02-09)
==================

- Added the ``Seismology.interact_echelle()`` method for creating interactive
  asteroseismic echelle diagrams. [#625]

- Added ``odd_mask`` and ``even_mask`` properties to ``FoldedLightCurve`` to
  make it easy to plot odd- and even-numbered transits. [#425]

- Fixed a bug which caused ``TargetPixelFile.interact()`` to raise a
  ``ValueError`` if the pixel file contained NaN flux values. [#679]

- Fixed minor issues in the tutorials. [#662, #683]



1.7.0 (2020-01-29)
==================

- Added a ``scale='linear'`` option to ``TargetPixelFile.interact()`` to show
  pixels using a linear stretch. The default is ``scale='log'``. [#664]

- Added a warning if ``SFFCorrector`` is used to correct TESS data. [#660]

- Added improved sigma-clipping inside ``RegressionCorrector``. [#654]

- Fixed a bug which caused ``LightCurve.show_properties()`` to raise a
  ``ValueError`` when the time format was not set. [#655]

- Fixed a bug which caused ``TargetPixelFile.interact()`` to crash if the
  pipeline aperture mask did not contain pixels. [#667]

- Fixed a bug which caused ``RegressionCorrector.correct()`` to crash if the
  input light curve contained flux uncertainties <= 0. [#668]



1.6.0 (2019-12-16)
==================

- Fixed a bug in ``tpf.to_lightcurve()`` which caused ``flux`` and ``flux_err``
  to be ``0`` instead of ``NaN`` for cadences with all-NaN pixels. [#651]

- Added a new TESS data anomaly flag (bit 13 / value 4096) which was introduced
  in Sector 14 to mark cadences affected by strong scattered light.  Compared
  to the original stray light flag (bit 12), this flag is set automatically by
  the pipeline based on background level thresholds. [#652]

- Changed the requirements to make ``fbpca`` a required dependency, because
  it allows ``DesignMatrix.pca()`` to be faster and more robust. [#653]



1.5.2 (2019-12-05)
==================

- Fixed a bug introduced in v1.5.0 which caused an ``ImportError`` related to
  ``astropy.stats.calculate_bin_edges`` to be raised when a user has an older
  version of AstroPy installed (version <3.1 or <2.10). [#644]

- Fixed a bug which caused the positions of stars in ``tpf.interact_sky()`` to
  be off by one pixel. [#638]



1.5.1 (2019-11-22)
==================

- Fixed a bug introduced in Lightkurve v1.5 which caused ``import lightkurve``
  on Mac OSX to automatically select the Matplotlib Agg backend. [#640]



1.5.0 (2019-11-20)
==================

- Changed the representation of ``SearchResult`` objects to make it easier to
  see at a glance which quarter/campaign/sector a result belongs to. [#632]

- Added ``mission``, ``sector``, ``camera``, and ``ccd`` properties to
  ``TessLightCurveFile`` for consistency with ``TessTargetPixelFile``. [#633]

- Added the ``bins`` argument to ``LightCurve.bin()`` to enable custom binning
  by specifying the bin edges or the total number of bins. [#629]

- Added ``transform_func`` & ``ylim_func`` keywords to ``interact()`` to
  support user-defined light curve transformations and y-axis limits. [#600]

- Added ``to_stringray()`` and ``from_stingray()`` to ``LightCurve`` to enable
  interoperability with the `Stingray <https://stingraysoftware.github.io/>`_
  spectral timing package. [#567]

- Added an `ax` (axes) keyword to ``Seismology.plot_echelle()`` to enable
  Echelle diagrams to be plotted into an existing Matplotlib figure. [#635]



1.4.1 (2019-11-18)
==================

- Fixed a bug which caused ``search_targetpixelfile`` and
  ``search_lightcurvefile`` to raise an ``IndexError`` if the sector keyword
  was passed and the target was observed by both TESS & Kepler. [#631]



1.4.0 (2019-11-12)
==================

- Added the generic ``RegressionCorrector`` and ``DesignMatrix`` classes which
  provide a user-friendly way to use linear regression to remove background or
  systematic noise components from light curves. [#613]

- Refactored the ``SFFCorrector`` class to use the new ``RegressionCorrector``,
  which deprecated the ``polyorder`` keyword in favor of ``degree``.
  [#613, #616, #617, #626]

- Changed the `tutorials index page <https://docs.lightkurve.org/tutorials>`_
  in the online docs to make the tutorials easier to navigate.

- Added a tutorial which demonstrates the use of Lightkurve's seismology module
  to measure the mass, radius, and surface gravity of a solar-like star. [#624]

- Changed ``SearchResult.download()`` to raise a more explicit ``HTTPError``
  exception when MAST's TESSCut service is overloaded and times out. [#627]



1.3.0 (2019-10-21)
==================

- Added a ``method="quadratic"`` option to ``tpf.estimate_centroids()`` which
  enables centroids to be estimated by fitting a bivariate polynomial to the
  3x3 pixel core of the PSF. The method can also be called as a standalone
  function via ``lightkurve.utils.centroid_quadratic()``. [#544, #610]

- Fixed a bug in ``Seismology.plot_echelle()`` which caused the Echelle diagram
  of a power spectrum to be rendered incorrectly. [#602]

- Fixed a bug which caused ``lightkurve.utils`` to be incorrectly resolved to
  ``lightkurve.seismology.utils``. [#606]

- Changed ``bkjd_to_astropy_time()`` and ``btjd_to_astropy_time()`` to accept
  a single float and lists of floats in addition to numpy arrays. [#608]

- Improved support for creating a ``LombScarglePeriodogram`` with an unevenly
  sampled grid in frequency space. [#614]



1.2.0 (2019-10-01)
==================

- Added ``flux_unit`` and ``flux_quantity`` properties to the ``LightCurve``
  class to enable users to keep track of a light curve's flux units. [#591]

- Changed the default behavior of ``LightCurve.plot()`` to use ``normalize=False``,
  ie. plots now display a light curve in its intrinsic units by default. [#591]

- Added an optional ``unit`` argument to ``LightCurve.normalize()`` to make it
  convenient to obtain a relative light curve in percent (``unit='percent'``),
  parts per thousand (``unit='ppt'``) or parts per million (``unit='ppm'``). [#591]

- Changed ``LombScarglePeriodogram.from_lightcurve()`` to not normalize the
  input light curve by default. [#591]

- Changed ``LightCurve.normalize()`` to emit a warning if the light curve
  appears to be zero-centered. [#589]

- Fixed an issue which caused the search functions to be incompatible with the
  latest version of astroquery (v0.3.10). [#598]

- Added support for performing mathematical operations involving ``LightCurve``
  objects, e.g. two ``LightCurve`` objects can now be added together. [#532]

- Updated the online tutorials (https://docs.lightkurve.org/tutorials) to
  take all recent Lightkurve API changes into account. [#596]



1.1.1 (2019-08-19)
==================

Lightkurve v1.1.1 is a bugfix release which includes the following changes:

- Changed ``search_targetpixelfile()`` and ``search_lightcurvefile()`` to emit a
  helpful warning if an ambigous target identifier is used, i.e. if a number is
  entered in the range where the K2 EPIC and TESS TIC catalogs overlap. [#558]

- Changed ``TargetPixelFile.plot()`` to always display the cadence number. [#562]

- Changed ``TargetPixelFile.interact()`` to store light curves created using the
  tool in the ``SAP_FLUX`` column rather than the ``FLUX`` column of the new
  light curve file, for consistency with pipeline products. [#559]

- Added ``scatter()`` and ``errorbar()`` methods to the ``LightCurveFile`` class
  to make it consistent with the ``LightCurve`` class. [#382]

- Fixed a bug in ``KeplerTargetPixelFile.from_fits_images()`` to ensure the
  correct pixels are selected in cutout mode. [#571]

- Fixed a series of minor documentation and code quality issues to enable
  Lightkurve to receive the "code quality A" certification by codacy.com.
  [#557, #560, #564, #565, #566, #568, #573, #574, #575]



1.1.0 (2019-07-19)
==================

- Added the ``lightkurve.seismology`` sub-package which enables quick-look
  asteroseismic quantities to be extracted from ``Periodogram`` objects. [#496]

- Added the ``stitch()`` method to ``LightCurveCollection`` and ``LightCurveFileCollection``
  to enable multi-sector/multi-quarter data to be combined more easily. [#548]

- Improved the ``LightCurve.fill_gaps()`` method to fill gaps in a light curve
  with Gaussian noise proportional to the light curve's CDPP. [#548]

- Added the ``TargetPixelFile.cutout()`` method which enables smaller Target
  Pixel Files to be extracted from larger ones. [#537]

- Added a ``pld_aperture_mask`` argument to ``PLDCorrector.correct()`` to enable
  users to select the pixels used for creating the PLD basis vectors. [#523]

- Added a new unit test module (test_synthetic_data.py) which utilizes
  synthetic Target Pixel Files to validate Lightkurve features. [#534]

- Added extra ``log.debug`` messages to ``lightkurve.search`` to enable users
  to track the status of search and download operations. [#547]

- Added several new usage examples to the docstrings of functions. [#516]

- Removed seven methods which had been deprecated prior to v1.0: [#515]
  * removed ``lc.cdpp()`` in favor of ``lc.estimate_cdpp()``;
  * removed ``lc.correct()`` in favor of ``lc.to_corrector().correct()``;
  * removed ``lcf.from_fits()`` in favor of ``lightkurve.open()``;
  * removed ``tpf.from_fits()`` in favor of ``lightkurve.open()``;
  * removed ``lcf.from_archive()`` in favor of ``search_lightcurvefile()``;
  * removed ``tpf.from_archive()`` in favor of ``search_targetpixelfile()``;
  * removed ``tpf.centroids()`` in favor of ``tpf.estimate_centroids()``.

- Moved the ``Corrector`` systematics removal classes into their own
  sub-package, named ``lightkurve.correctors``. [#519]

- Fixed a bug which prevented ``lightkurve.open()`` from raising a
  ``FileNotFoundError`` when a file does not exist. [#540]

- Fixed a bug which caused ``BoxLeastSquaresPeriodogram`` to ignore the
  ``period`` parameter. [#514]

- Fixed a bug which prevented the ``t0`` argument of ``lc.fold()`` from being
  an AstroPy Quantity object. [#521]



1.0.1 (2019-05-20)
==================

This is a minor bugfix release containing the following improvements:

- Fixed minor bugs in ``PLDCorrector.correct()`` [#498],
  ``TargetPixelFile.create_threshold_mask()`` [#502],
  and ``LightCurve.bin()`` [#503].

- Ensure users are alerted if a large number of cadences are masked out by
  ``quality_bitmask`` when opening data products. [#495]

- ``CBVCorrector`` now accepts a ``KeplerLightCurve`` as input. [#504]

- The ``lightkurve.search`` functions now provide a more helpful error message
  if the download cache contains a corrupt file. [#512]

- Switched continuous integration from Travis/Appveyor to Azure. [#497]



1.0.0 (2019-04-08)
==================

This is the first stable release of Lightkurve.  It was prepared with the help
of 45 contributors!

This release contains major changes to the ``LombScarglePeriodogram`` class:

- Changed the default behavior of ``LombScarglePeriodogram.from_lightcurve()``
  to use ``normalization='amplitude'`` and ``oversample_factor=5`` (the previous
  defaults were ``normalization='psd'`` and ``oversample_factor=1``).
  The docstring has been expanded to help users understand these options. [#491]

- Added a ``LightkurveWarning`` to alert users of the changes to the default
  behavior. [#493]

- Deprecated the ``min_frequency``/``max_frequency`` arguments in favor of
  ``minimum_frequency``/``maximum_frequency`` to be consistent with the other
  Periodogram classes. [#478]

- Likewise, deprecated the ``min_period``/``max_period`` arguments in favor of
  ``minimum_period``/``maximum_period`` to be consistent with the other
  Periodogram classes. [#478]

Other changes are:

- Improved ``PLDCorrector`` to be more robust against the presence of NaNs.
  [#479, #488]

- Improved ``search_tesscut`` to avoid crashing in the event of an empty search
  result, and to ensure that the files it returns carry the search string as
  the ``targetid`` attribute. [#475, #477]

- Various minor bug fixes. [#488, #490, #494]



1.0b30 (2019-03-27)
===================

- Significantly improved the performance of the ``PLDCorrector`` feature for
  systematics removal. [#470]

- Improved the normalization of the result returned by
  ``Periodogram.smooth(method='logmedian')``. [#453]

- Improved the visualization of NaN values in ``TargetPixelFile.plot()``. [#455]

- Various minor bug fixes. [#448, #450, #463, #471]



1.0b29 (2019-02-14)
===================

- The ``search_tesscut(...).download()`` feature now supports downloading
  rectangular TESS FFI cut-outs. It previously only supported squares. [#441]

- Fixed a bug which prevented ``search_tesscut(...).download_all()`` from
  downloading all sectors. [#440]

- Minor bug fixes and performance improvements. [#439, #446]



1.0b28 (2019-02-09)
===================

Changes
-------

- Simplified the installation of Lightkurve by turning several packages into
  optional rather than required dependencies (``celerite``, ``pybind``,
  ``scikit-learn``, and ``bokeh``). [#436]

- Added ``search_tesscut()``: an easy interface to access data produced using
  the `MAST TESSCut service <https://mast.stsci.edu/tesscut/>`_. This service
  extracts Target Pixel Files (TPFs) from TESS Full Frame Images (FFIs). [#418]

- Added ``TargetPixelFile.interact_sky()``: an interactive Bokeh widget to
  overlay Gaia DR2 source positions on top of TPFs. [#124]

- Changed ``LightCurve.fold()``: the ``transit_midpoint`` parameter has been
  deprecated in favor of the ``t0`` parameter. [#419]

Bugfixes
--------

- Made ``BoxLeastSquaresPeriodogram`` robust against light curves that contain
  NaNs. [#432]

- ``TargetPixelFile.wcs`` now works for Target Pixel Files produced using the
  MAST TessCut service. [#434]



1.0b26 (2019-02-04)
===================

- Introduced a new layout for the
  `online documentation <https://docs.lightkurve.org>`_. [#360, #400, #406]

- Added ``LightCurve.interact_bls()``: an interactive Bokeh widget to find
  planets using the Box Least Squares (BLS) method. [#401]

- Added ``LombScarglePeriodogram`` and ``BoxLeastSquarePeriodogam`` sub-classes
  to distinguish periodograms generated using different methods. [#403]

- Added the ``PLDCorrector`` class to remove instrument systematics using the
  Pixel Level Decorrelation (PLD) method. [#305]

- Added the ``TargetPixelFile.to_corrector()`` convenience method to make
  systematics correction classes easy to access. [#305]

- Refactored ``SFFCorrector`` to make its API consistent with ``PLDCorrector``,
  and deprecated the ``LightCurve.correct()`` method in favor of
  ``LightCurve.to_corrector()``. [#408, #417]

- Made ``SFFCorrector`` robust against light curves that contain big gaps in
  time. [#414]

- Minor bug fixes. [#392, #397, #420]

- Increased the unit test coverage. [#387, #388]



1.0b25 (2018-12-14)
===================

- The ``TargetPixelFile.interact()`` bokeh app now includes a ``Save Lightcurve``
  button [#329].

- Fixed a minor bug in ``LightCurve.bin()`` [#377].



1.0b24 (2018-12-10)
===================

- Added support for TESS to ``search_targetpixelfile()`` and
  ``search_lightcurvefile()`` [#367].

- Added support for data generated by the
  `TESScut service <https://mast.stsci.edu/tesscut/>`_ [#369, #375].

- Removed "Impulsive outliers" from the default set of quality constraints
  applied to TESS data [#374].

- ``LightCurve.flatten()`` is now more robust against outliers [#372].

- ``LightCurve.fold()`` now takes a ``transit_midpoint`` parameter instead of
  the ``phase`` parameter [#361, #363].

- Various minor bugfixes [#372].



1.0b23 (2018-11-30)
===================

- ``TargetPixelFile.create_threshold_mask()`` now only returns one contiguous
  mask, which is configurable using the new ``reference_pixel`` argument [#345].

- ``TargetPixelFile.interact()``: now requires ``Bokeh v1.0`` or later [#355].

- ``utils.detect_filetype()`` automatically detects Kepler or TESS Target Pixel
  Files and Light Curve files [#340, #350, #356].

- ``LightCurve.estimate_cdpp()``: the argument ``sigma_clip`` was renamed into
  ``sigma`` [#359].

- Fixed minor bugs in ``LightCurve.to_pandas()`` [#343],
  ``LightCurve.correct()`` [#347], ``FoldedLightCurve.errorbar()`` [#352],
  ``LightCurve.fold()`` [#353].

- Documentation improvements [#344, #358].

- Increased the unit test coverage [#351].



1.0b22 (2018-11-17)
===================

- ``lightkurve.open()`` was added to provide a single function to read in any
  light curve or target pixel file from Kepler or TESS and return the appropriate
  object [#317].

- The ``from_fits()`` methods have been deprecated in favor of
  ``lightkurve.open()`` [#336].

- The ``lightkurve.mast`` module has been removed in favor of the new
  ``lightkurve.search`` module.

- Various small bugfixes, speed-ups, and documentation improvements
  [#314, #315, #322, #323, #325, #331, #334, #335].



1.0b21 (2018-10-29)
===================

- The ``from_archive()`` methods of ``KeplerTargetPixelFile`` and
  ``KeplerLightCurveFile`` have been deprecated in favor of the new
  ``search_targetpixelfile()`` and ``search_lightcurvefile()`` functions.
  These allow users to inspect the results of their queries and offer more
  powerful features, e.g. cone-searches.  If you are currently using
  ``tpf = KeplerTargetPixelFile.from_archive("objectname")``, please start
  using ``tpf = search_targetpixelfile("objectname").download()`` instead.

- ``TargetPixelFile`` objects can now be indexed and sliced. [#308]

- The default number of ``windows`` used by the SFF systematics removal
  algorithm has been changed from 1 to 10. [#312]

- Various small bug fixes and unit test improvements.



1.0b20 (2018-10-16)
===================

- We adopted a rule that all method names must include a verb, and all class
  properties must be a noun [#286].  As a result, we renamed the following methods:

  * ``LightCurve.cdpp()`` is now ``LightCurve.estimate_cdpp()``

  * ``LightCurve.periodogram()`` is now ``LightCurve.to_periodogram()``

  * ``LichtCurve.properties()`` is now ``LightCurve.show_properties()``

  * ``TargetPixelFile.aperture_photometry()`` is now
    ``TargetPixelFile.extract_aperture_photometry()``

  * ``TargetPixelFile.centroids()`` is now ``TargetPixelFile.estimate_centroids()``

  * ``TargetPixelFile.header()`` is now a property.

- Added ``Periodogram.smooth()`` [#288].

- ``Periodogram.estimate_snr()`` was renamed to ``Periodogram.p.flatten()`` [#290].

- Lightkurve can now read in light curve files produced using
  ``LightCurveFile.to_fits()`` [#297].



1.0b19 (2018-10-10)
===================

- The ``Periodogram`` class has been refactored;

- The ``LightCurve.remove_outliers()`` method now accepts ``sigma_lower`` and
  ``sigma_upper`` parameters.
