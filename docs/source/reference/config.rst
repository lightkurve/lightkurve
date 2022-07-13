.. _api.config:

=============
Configuration
=============
.. currentmodule:: lightkurve


``Lightkurve`` uses ``Astropy``'s configuration system for configurable parameters.

Users can set their defaults in their configuration file, defaulted at ``$HOME/.lightkurve/config/lightkurve.cfg``.

Furthermore, they can also change the values at runtime via `lightkurve.conf` object.

The remaining specifics can be found in `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html>`_.


Access configuration values
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: api/

  conf
  config.get_cache_dir
  config.get_config_dir


Default Cache Directory Migration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from ``Lightkurve`` version ``2.4.0``, the default cache directory is
at ``~/.lightkurve/cache`` . The data files cached at the legacy location,
``~/.lightkurve-cache``, will not be used.

A warning is issued if the legacy ``~/.lightkurve-cache`` directory still exists.

Migration suggestions for handling various scenarios:

* To use the existing data files cached, move all the contents under ``~/.lightkurve-cache``
  to ``~/.lightkurve/cache``, and remove ``~/.lightkurve-cache`` directory itself.

* If you need to use older version of ``Lightkurve``, e.g., because of the requirements
  of other packages / applications, you can:

  #. Keep the cache at the legacy location ``~/.lightkurve-cache``
  #. Instruct current ``Lightkurve`` to use the legacy location. In the user's ``lightkurve.cfg``, add:

      [config]

      cache_dir = /<your-home-directory>/.lightkurve-cache

  #. The warning will no long appear once a custom ```cache_dir``` is specified.

* To suppress the warning for any reason, you can set ``warn_legacy_cache_dir`` in the user's ``lightkurve.cfg``.

    [config]

    warn_legacy_cache_dir = False
