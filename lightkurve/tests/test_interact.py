"""Tests the features of the lightkurve.interact module."""
from astropy.utils.data import get_pkg_data_filename
import pytest

from ..targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from ..interact import *

filename_tpf_one_center = get_pkg_data_filename("data/test-tpf-non-zero-center.fits")


def test_malformed_notebook_url():
    """Test if malformed notebook_urls raise proper exceptions."""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    with pytest.raises(ValueError) as exc:
        tpf.interact(notebook_url='')
    assert('Empty host value' in exc.value.args[0])
    with pytest.raises(AttributeError) as exc:
        tpf.interact(notebook_url=None)
    assert('object has no attribute' in exc.value.args[0])


def test_graceful_exit_outside_notebook():
    """Test if running interact outside of a notebook does fails gracefully."""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    result = tpf.interact()
    assert(result is None)


def test_custom_lc():
    """Can we provide a custom lightcurve to show?"""
    for tpf in [KeplerTargetPixelFile(filename_tpf_one_center),
                TessTargetPixelFile(filename_tpf_one_center)]:
        tpf.interact(lc=tpf.to_lightcurve().flatten())


def test_interact_functions():
    """Do the helper functions in the interact module run without syntax error?"""
    tpf = KeplerTargetPixelFile(filename_tpf_one_center)
    lc = tpf.to_lightcurve()
    tpf_source = prepare_tpf_datasource(tpf)
    lc_source = prepare_lightcurve_datasource(lc)
    get_lightcurve_y_limits(lc_source)
    make_lightcurve_figure_elements(lc, lc_source)
    make_tpf_figure_elements(tpf, tpf_source)
    show_interact_widget(tpf)
