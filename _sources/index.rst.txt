.. title:: Lightkurve docs

.. raw:: html

  <div class="container text-center" style="padding-top:1em; padding-bottom: 2em;">
    <h1 style="font-size: 4rem;">Lightkurve</h1>
    <p class="lead">
        A friendly package for Kepler & TESS time series analysis in Python.
    </p>
    <p>
        <a href="quickstart.html" class="btn btn-primary my-2" style="font-size: 1.25rem;">Quickstart →</a>
    </p>
  </div>

  <div class="container">
    <hr>
    <div class="row">
      <div class="col-lg-6">
        <p style="font-size: 1.2em; font-weight: 700;">
         Time domain astronomy made easy
        </p>
        <p>
          Lightkurve offers a user-friendly way
          to analyze time series data obtained by telescopes,
          in particular NASA’s Kepler and TESS exoplanet missions.
        </p>
        <p>
          Lightkurve aims to lower barriers, promote best practices, and reduce costs
          by providing high-quality <a href="api/index.html">API's</a> and
          <a href="tutorials/index.html">tutorials</a>
          accessible to everyone.
        </p>
      </div> 

      <div class="col-lg-6">


.. code-block:: python

    import lightkurve as lk

    pixels = lk.search_targetpixelfile("Kepler-10").download()
    pixels.plot()

    lightcurve = pixels.to_lightcurve()
    lightcurve.plot()

    exoplanet = lightcurve.flatten().fold(period=0.838)
    exoplanet.plot()


.. raw:: html

      </div>
    </div>
  </div>
