1.1.0 (unreleased)
==================

- Added the `tpf.cutout()` method which enables smaller Target Pixel Files to
  be extracted from larger ones. [#537]

- Added the `pld_aperture_mask` argument to `PLDCorrector.correct()` to enable
  users to select the pixels used for creating the PLD basis vectors. [#523]

- Added a new unit test module (test_synthetic_data.py) which utilizes
  synthetic Target Pixel Files to validate Lightkurve features. [#534]

- Added extra `log.debug` messages to `lightkurve.search` to enable users
  to track the status of search and download operations. [#547]

- Added several new usage examples to the docstrings of functions. [#516]

- Removed seven methods which had been deprecated prior to v1.0: [#515]
  * removed `lc.cdpp()` in favor of `lc.estimate_cdpp()`;
  * removed `lc.correct()` in favor of `lc.to_corrector().correct()`;
  * removed `lcf.from_fits()` in favor of `lightkurve.open()`;
  * removed `tpf.from_fits()` in favor of `lightkurve.open()`;
  * removed `lcf.from_archive()` in favor of `search_lightcurvefile()`;
  * removed `tpf.from_archive()` in favor of `search_targetpixelfile()`;
  * removed `tpf.centroids()` in favor of `tpf.estimate_centroids()`.

- Moved the `Corrector` systematics removal classes into their own sub-package,
  named `lightkurve.correctors`. [#519]

- Fixed a bug which prevented `lightkurve.open()` from raising a
  `FileNotFoundError` when a file does not exist. [#540]

- Fixed a bug which caused `BoxLeastSquaresPeriodogram` to ignore the `period`
  parameter. [#514]

- Fixed a bug which prevented the `t0` argument of `lc.fold()` from being an
  AstroPy Quantity object. [#521]



1.0.1 (2019-05-20)
==================

This is a minor bugfix release containing the following improvements:

- Fixed minor bugs in ``PLDCorrector.correct()`` [#498],
  ``TargetPixelFile.create_threshold_mask()`` [#502],
  and ``LightCurve.bin()`` [#503].

- Ensure users are alerted if a large number of cadences are masked out by
`quality_bitmask` when opening data products. [#495]

- `CBVCorrector` now accepts a `KeplerLightCurve` as input. [#504]

- The `~lightkurve.search` functions now provide a more helpful error message
if the download cache contains a corrupt file. [#512]

- Switched continuous integration from Travis/Appveyor to Azure pipelines. [#497]



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
  ``online documentation <https://docs.lightkurve.org>``_. [#360, #400, #406]

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
