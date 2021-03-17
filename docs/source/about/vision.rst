.. _vision:

================================
Vision of the Lightkurve project
================================

This page summarizes the vision of the Lightkurve project.
This document aims to be a reference to aid making decisions on the features, design, or future of the package.
This vision is not set in stone, anyone is invited to open a GitHub issue to discuss or revise any aspects of it.


Goals
-----

Lightkurve aims to support the analysis of time series data on planets, stars, and galaxies
obtained by telescopes which collect images in visible or infrared light (e.g., NASA's Kepler, TESS, and Roman missions).
Specifically, Lightkurve aims to empower its users with a robust set of re-usable building blocks which enable
the development of more specialized toolkits, pipelines, and hand-tailored analyses.

Crucially, Lightkurve should not preclude any other package from existing, because there will always be a need for more
specialized/complex tools for analyzing optical time series data.
Instead, Lightkurve aims to avoid duplication of efforts across tools and analyses
to achieve the following goals:

* lower the barrier to time domain astronomy (including citizen science);
* promote and document best practices in the field;
* reduce the costs of implementing custom analyses;
* improve scientific fidelity across different tools;
* improve consistency across different pipelines.

To achieve these goals, it is essential that Lightkurve adopts open source development best
practices, including careful peer review of changes, thorough documentation, and continuous testing of the code.


What features can be included in Lightkurve?
--------------------------------------------

Lightkurve focuses on providing robust open source implementations
of re-usable building blocks.
As a general rule, Lightkurve should only include features which...

* are widely adopted across the field;
* are applicable across a wide range of data;
* serve multiple scientific use cases;
* are not particularly novel or controversial;
* are well-tested and well-documented.

Examples include:

* obtaining Kepler and TESS data programmatically from their data archives;
* reading, visualizing, and interacting with Kepler and TESS pipeline products;
* extracting light curves from pixel data using custom aperture masks;
* executing common light curve operations (e.g. folding, binning, removing outliers);
* extracting different types of periodograms; 
* removing systematics using tunable implementations of the most common systematic removal strategies.

Lightkurve does not intend to be a full pipeline or one-stop shop analysis toolkit on its own,
instead it seeks to empower others to build more specialized tools on top of it.


What should probably not be included in Lightkurve?
---------------------------------------------------

Lightkurve cannot include every possible feature for every imaginable use case.
This would make Lightkurve's maintenance and development prohibitively expensive.

Features which are likely outside of Lightkurve's scope include:

* end-to-end analysis pipelines (e.g., planet or supernova discovery pipelines);
* brand new methods which have not yet been adopted widely;
* tools to analyze unique edge cases or obscure data;
* interactive graphical user interfaces;
* tools to work with radio or X-ray time series data.

If in doubt, please open a GitHub issue to discuss!


How does Lightkurve relate to AstroPy TimeSeries?
-------------------------------------------------

Lighkurve is built on top of the `AstroPy project <https://www.astropy.org/>`_ and could not exist without it.
For example, Lightkurve's ``LightCurve`` object is an extension of the AstroPy ``TimeSeries`` object,
which in turn is a sub-class of AstroPy's ``Table`` object.

An AstroPy ``TimeSeries`` object is a table which is guaranteed to have a special ``time`` column,
which is always the first column and guaranteed to be an ``astropy.time.Time`` object.
``LightCurve`` extends this class by adding special ``flux`` and ``flux_err`` columns, which are always the second and third column. 
These dedicated columns enable us to provide extra methods which are specific to the manipulation of brightness data.
For example, methods which are currently unique to ``LightCurve`` objects include ``plot()``, ``normalize()``, ``fill_gaps()``,
``flatten()``, ``remove_outliers()``, ``remove_nans()``, ``estimate_cdpp()``, ``plot_river()``, ``to_seismology()``, etc.

Some of these methods may eventually be migrated to AstroPy if they are sufficiently generic and mature
to meet the requirements of the AstroPy project, while many Kepler/TESS-specific methods will likely remain
only in Lightkurve to avoid burdening AstroPy with mission-specific features.
