"""Tests the features of the lightkurve.interact module."""
from astropy.utils.data import get_pkg_data_filename
import pytest
import warnings
import sys

from .. import LightkurveWarning
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile

example_tpf = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")
TABBY_TPF = ("https://archive.stsci.edu/missions/kepler/target_pixel_files"
             "/0084/008462852/kplr008462852-2011073133259_lpd-targ.fits.gz")


def test_bokeh_import_error(caplog):
    """If bokeh is not installed (optional dependency),
    is a friendly error message printed?"""
    try:
        import bokeh
    except ImportError:
        tpf = TessTargetPixelFile(example_tpf)
        tpf.interact()
        assert "requires the `bokeh` Python package" in caplog.text


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_malformed_notebook_url():
    """Test if malformed notebook_urls raise proper exceptions."""
    import bokeh
    tpf = TessTargetPixelFile(example_tpf)
    with pytest.raises(ValueError) as exc:
        tpf.interact(notebook_url='')
    assert('Empty host value' in exc.value.args[0])
    with pytest.raises(AttributeError) as exc:
        tpf.interact(notebook_url=None)
    assert('object has no attribute' in exc.value.args[0])


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_graceful_exit_outside_notebook():
    """Test if running interact outside of a notebook does fails gracefully."""
    import bokeh
    tpf = TessTargetPixelFile(example_tpf)
    result = tpf.interact()
    assert(result is None)


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_custom_aperture_mask():
    """Can we provide a custom lightcurve to show?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(TABBY_TPF),
                TessTargetPixelFile(example_tpf)]
    import bokeh
    for tpf in tpfs:
        mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
        tpf.interact(aperture_mask=mask)
        mask = None
        tpf.interact(aperture_mask=mask)
        mask = 'threshold'
        tpf.interact(aperture_mask=mask)


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_custom_exported_filename():
    """Can we provide a custom lightcurve to show?"""
    import bokeh
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(TABBY_TPF),
                TessTargetPixelFile(example_tpf)]
    for tpf in tpfs:
        tpf.interact(exported_filename='demo.fits')
        tpf[0:2].interact()
        tpf[0:2].interact(exported_filename='string_only')
        tpf[0:2].interact(exported_filename='demo2.FITS')
        tpf[0:2].interact(exported_filename='demo3.png')
        tpf[0:2].interact(exported_filename='')
        tpf.interact(exported_filename=210690913)
        mask = tpf.time == tpf.time
        tpf[mask].interact()


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_max_cadences():
    """Can we provide a custom lightcurve to show?"""
    import bokeh
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(TABBY_TPF),
                TessTargetPixelFile(example_tpf)]
    for tpf in tpfs:
        with pytest.raises(RuntimeError) as exc:
            tpf.interact(max_cadences=2)
            assert('Interact cannot display more than' in exc.value.args[0])


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_interact_functions():
    """Do the helper functions in the interact module run without syntax error?"""
    import bokeh
    from ..interact import (prepare_tpf_datasource, prepare_lightcurve_datasource,
                            get_lightcurve_y_limits, make_lightcurve_figure_elements,
                            make_tpf_figure_elements, show_interact_widget)
    tpf = TessTargetPixelFile(example_tpf)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    lc = tpf.to_lightcurve(aperture_mask=mask)
    lc_source = prepare_lightcurve_datasource(lc)
    get_lightcurve_y_limits(lc_source)
    make_lightcurve_figure_elements(lc, lc_source)
    make_tpf_figure_elements(tpf, tpf_source)
    show_interact_widget(tpf)


@pytest.mark.skipif('bokeh' not in sys.modules, reason="requires bokeh")
def test_interact_sky_functions():
    """Do the helper functions in the interact module run without syntax error?"""
    import bokeh
    from ..interact import (prepare_tpf_datasource, make_tpf_figure_elements,
                            add_gaia_figure_elements)
    tpf = TessTargetPixelFile(example_tpf)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig1, slider1 = make_tpf_figure_elements(tpf, tpf_source)
    add_gaia_figure_elements(tpf, fig1)
    add_gaia_figure_elements(tpf, fig1, magnitude_limit=22)
