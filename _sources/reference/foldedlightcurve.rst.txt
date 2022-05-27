.. _api.foldedlightcurve:

================
FoldedLightCurve
================
.. currentmodule:: lightkurve

The `FoldedLightCurve` class extends a standard `LightCurve` object by having
the ``time`` column represent phase values.
This allows a `FoldedLightCurve` to offer a few features that are specific to
phase-folded time series data, namely the `~FoldedLightCurve.phase`, `~FoldedLightCurve.odd_mask`,
and `~FoldedLightCurve.even_mask` attributes.
The class also overrides the plotting methods to provide defaults that are
suitable for plotting phase-folded data.


Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   FoldedLightCurve


Extra attributes
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   FoldedLightCurve.phase
   FoldedLightCurve.cycle
   FoldedLightCurve.odd_mask
   FoldedLightCurve.even_mask


Modified plotting methods
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  FoldedLightCurve.plot
  FoldedLightCurve.scatter
  FoldedLightCurve.errorbar
  FoldedLightCurve.plot_river

