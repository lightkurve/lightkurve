.. _citing:

====================================
Citing Lightkurve & its dependencies
====================================


If you use Lightkurve for work or research presented in a publication, we
request the following acknowledgment or citation:

.. code-block:: text

  This research made use of Lightkurve, a Python package for Kepler and TESS data analysis (Lightkurve Collaboration, 2018).

where (Lightkurve Collaboration, 2018) is a citation to the ADS entry `2018ascl.soft12013L <http://adsabs.harvard.edu/abs/2018ascl.soft12013L>`_.
The recommended BibTeX entry for this citation is:

.. code-block:: latex


    @MISC{2018ascl.soft12013L,
       author = {{Lightkurve Collaboration} and {Cardoso}, J.~V.~d.~M. and 
                 {Hedges}, C. and {Gully-Santiago}, M. and {Saunders}, N. and 
                 {Cody}, A.~M. and {Barclay}, T. and {Hall}, O. and 
                 {Sagear}, S. and {Turtelboom}, E. and {Zhang}, J. and 
                 {Tzanidakis}, A. and {Mighell}, K. and {Coughlin}, J. and 
                 {Bell}, K. and {Berta-Thompson}, Z. and {Williams}, P. and 
                 {Dotson}, J. and {Barentsen}, G.},
        title = "{Lightkurve: Kepler and TESS time series analysis in Python}",
     keywords = {Software, NASA},
    howpublished = {Astrophysics Source Code Library},
         year = 2018,
        month = dec,
    archivePrefix = "ascl",
       eprint = {1812.013},
       adsurl = {http://adsabs.harvard.edu/abs/2018ascl.soft12013L},
    }

In addition, you may elect to cite a specific version of Lightkurve using the `version-specific DOI <https://doi.org/10.5281/zenodo.1181928>`_ provided by Zenodo.

Citing dependencies
-------------------

Lightkurve was built on top of a number of powerful libraries,
including `NumPy <https://www.numpy.org/>`_, `SciPy <https://scipy.org>`_, and `Matplotlib <https://matplotlib.org/>`_.
We strongly encourage you to cite these packages as well.
In particular, we request that all astronomy publications cite the relevant
astronomy packages, which include:

* `astropy <https://astropy.org>`_ (see `citation instructions <https://www.astropy.org/acknowledging.html>`_) enabled all features.
* `astroquery <https://astroquery.readthedocs.io>`_ (see `citation instructions <https://github.com/astropy/astroquery#citing-astroquery>`_) enabled all `~lightkurve.search` functions.
* `tesscut <https://mast.stsci.edu/tesscut/>`_ (see `citation instructions <https://ascl.net/code/v/2239>`_) enabled `~lightkurve.search.search_tesscut`.
* `celerite <https://celerite.readthedocs.io>`_ (see `citation instructions <https://celerite.readthedocs.io/en/stable/#license-attribution>`_) enabled `~lightkurve.correctors.PLDCorrector`.

If your package is missing from the list above, please `open a pull request <https://github.com/KeplerGO/lightkurve>`_ to add it.
