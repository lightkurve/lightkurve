.. _api.io:

=====================
Reading Data Products
=====================
.. currentmodule:: lightkurve.io


.. autosummary::
  :toctree: api/

    read

In general, users only need to call the `read` function, also available as
``lightkurve.read``.  This function will auto-detect the type of data product
being opened and pass it on to a product-specific reader function.

Below we list the product-specific reader functions.  These functions
are not intended to be called directly because it is much easier to
simply call the `read` function.  We include the functions here because
their documentation lists the optional parameters that can be passed to
the generic `read()` function, and because they document information
pertaining to specific products.


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
    eleanor.read_eleanor_lightcurve
    pathos.read_pathos_lightcurve
    qlp.read_qlp_lightcurve
    tasoc.read_tasoc_lightcurve
