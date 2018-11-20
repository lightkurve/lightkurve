"""Tests the features of the lightkurve.interact module."""
from astropy.utils.data import get_pkg_data_filename
import pytest
import warnings

from .. import LightkurveWarning
from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile

example_tpf = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")


def test_bokeh_import_error(caplog):
    """If bokeh is not installed (optional dependency),
    is a friendly error message printed?"""
    try:
        import bokeh
    except ImportError:
        tpf = TessTargetPixelFile(example_tpf)
        tpf.interact()
        assert "requires the `bokeh` package" in caplog.text


def test_malformed_notebook_url():
    """Test if malformed notebook_urls raise proper exceptions."""
    try:
        import bokeh
        tpf = TessTargetPixelFile(example_tpf)
        with pytest.raises(ValueError) as exc:
            tpf.interact(notebook_url='')
        assert('Empty host value' in exc.value.args[0])
        with pytest.raises(AttributeError) as exc:
            tpf.interact(notebook_url=None)
        assert('object has no attribute' in exc.value.args[0])
    except ImportError:
        # bokeh is an optional dependency
        pass


def test_graceful_exit_outside_notebook():
    """Test if running interact outside of a notebook does fails gracefully."""
    try:
        import bokeh
        tpf = TessTargetPixelFile(example_tpf)
        result = tpf.interact()
        assert(result is None)
    except ImportError:
        # bokeh is an optional dependency
        pass


def test_custom_aperture_mask():
    """Can we provide a custom lightcurve to show?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(example_tpf),
                TessTargetPixelFile(example_tpf)]
    try:
        import bokeh
        with warnings.catch_warnings():
            # Ignore the "TELESCOP is not equal to TESS" warning
            warnings.simplefilter("ignore", LightkurveWarning)
            tpfs = [KeplerTargetPixelFile(example_tpf),
                    TessTargetPixelFile(example_tpf)]
        for tpf in tpfs:
            mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
            tpf.interact(aperture_mask=mask)
    except ImportError:
        # bokeh is an optional dependency
        pass


def test_custom_exported_filename():
    """Can we provide a custom lightcurve to show?"""
    try:
        import bokeh
        with warnings.catch_warnings():
            # Ignore the "TELESCOP is not equal to TESS" warning
            warnings.simplefilter("ignore", LightkurveWarning)
            tpfs = [KeplerTargetPixelFile(example_tpf),
                    TessTargetPixelFile(example_tpf)]
        for tpf in tpfs:
            tpf.interact(exported_filename='demo.fits')
    except ImportError:
        # bokeh is an optional dependency
        pass

def test_max_cadences():
    """Can we provide a custom lightcurve to show?"""
    try:
        import bokeh
        with warnings.catch_warnings():
            # Ignore the "TELESCOP is not equal to TESS" warning
            warnings.simplefilter("ignore", LightkurveWarning)
            tpfs = [KeplerTargetPixelFile(example_tpf),
                    TessTargetPixelFile(example_tpf)]
        for tpf in tpfs:
            with pytest.raises(RuntimeError) as exc:
                tpf.interact(max_cadences=2)
                assert('Interact cannot display more than' in exc.value.args[0])
    except ImportError:
        # bokeh is an optional dependency
        pass


def test_interact_functions():
    """Do the helper functions in the interact module run without syntax error?"""
    try:
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
    except ImportError:
        # bokeh is an optional dependency
        pass
