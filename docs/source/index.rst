.. title:: Lightkurve docs

.. container:: lead

    Lightkurve

    A friendly package for Kepler & TESS time series analysis in Python.

    .. raw:: html

        <a href="quickstart.html" class="btn btn-primary">Quickstart →</a>


.. raw:: html

  <div class="container">
    <hr>
    <div class="row">
      <div class="col-lg-6">
        <p style="font-size: 1.2em; font-weight: 700;">
         Time domain astronomy made easy for all
        </p>
        <p>
          Lightkurve offers a user-friendly way
          to analyze time series data obtained by telescopes,
          in particular NASA’s Kepler and TESS exoplanet missions.
        </p>
        <p>
          Lightkurve aims to lower barriers, promote best practices, reduce costs,
          and improve scientific fidelity
          by providing accessible Python <a href="api/index.html">tools</a> and
          <a href="tutorials/index.html">tutorials</a>.
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
