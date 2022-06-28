.. _api.io:

=====================
Reading Data Products
=====================
.. currentmodule:: lightkurve.io


.. autosummary::
  :toctree: api/

    read

In general, users only need to call `read` function, also available as ``lightkurve.read``,
to read all supported Kepler / TESS data products.

The data product-specific functions below **should not** be called directly.
They do, however, contain information pertaining to specific products, as well as
optional parameters that can be passed to the generic `read()`.


Kepler Data Products
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

    kepler.read_kepler_lightcurve
    everest.read_everest_lightcurve
    kepseismic.read_kepseismic_lightcurve
    k2sff.read_k2sff_lightcurve


TESS Data Products
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

    tess.read_tess_lightcurve
    cdips.read_cdips_lightcurve
    eleanorlite.read_eleanorlite_lightcurve
    pathos.read_pathos_lightcurve
    qlp.read_qlp_lightcurve
    tasoc.read_tasoc_lightcurve
