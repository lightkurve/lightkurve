.. _api.targetpixelfile:

===============
TargetPixelFile
===============
.. currentmodule:: lightkurve

The `lightkurve.targetpixelfile` module provides classes which represent FITS files
that store the original pixel data (images) obtained by the Kepler or TESS telescopes.
These classes provide methods to visualize these data and extract custom light curves.

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   KeplerTargetPixelFile
   TessTargetPixelFile

.. autosummary::
   :toctree: api/

   KeplerTargetPixelFile.hdu
   KeplerTargetPixelFile.ra
   KeplerTargetPixelFile.dec
   KeplerTargetPixelFile.column
   KeplerTargetPixelFile.row
   KeplerTargetPixelFile.pos_corr1
   KeplerTargetPixelFile.pos_corr2
   KeplerTargetPixelFile.pipeline_mask
   KeplerTargetPixelFile.shape
   KeplerTargetPixelFile.time
   KeplerTargetPixelFile.cadenceno
   KeplerTargetPixelFile.nan_time_mask
   KeplerTargetPixelFile.flux
   KeplerTargetPixelFile.flux_err
   KeplerTargetPixelFile.flux_bkg
   KeplerTargetPixelFile.flux_bkg_err
   KeplerTargetPixelFile.quality
   KeplerTargetPixelFile.wcs
   KeplerTargetPixelFile.get_coordinates
   KeplerTargetPixelFile.to_lightcurve
   KeplerTargetPixelFile.extract_aperture_photometry
   KeplerTargetPixelFile.extract_prf_photometry
   KeplerTargetPixelFile.get_model
   KeplerTargetPixelFile.create_threshold_mask
   KeplerTargetPixelFile.estimate_background
   KeplerTargetPixelFile.estimate_centroids
   KeplerTargetPixelFile.query_solar_system_objects
   KeplerTargetPixelFile.plot
   KeplerTargetPixelFile.to_fits
   KeplerTargetPixelFile.interact
   KeplerTargetPixelFile.interact_sky
   KeplerTargetPixelFile.to_corrector
   KeplerTargetPixelFile.cutout
   KeplerTargetPixelFile.from_fits_images
   KeplerTargetPixelFile.plot_pixels
   KeplerTargetPixelFile.get_header
   KeplerTargetPixelFile.get_keyword
   KeplerTargetPixelFile.animate
