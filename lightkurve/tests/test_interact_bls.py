"""Tests the features of the lightkurve.interact_bls module."""
import pytest

from ..lightcurvefile import KeplerLightCurveFile, TessLightCurveFile
from .test_lightcurve import KEPLER10, TESS_SIM

bad_optional_imports = False
try:
    import bokeh
    from astropy.stats.bls import BoxLeastSquares
except:
    bad_optional_imports = True


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_malformed_notebook_url():
    """Test if malformed notebook_urls raise proper exceptions."""
    lcf = KeplerLightCurveFile(KEPLER10)
    lc = lcf.PDCSAP_FLUX.normalize().remove_nans().flatten()
    with pytest.raises(ValueError) as exc:
        lc.interact_bls(notebook_url='')
    assert('Empty host value' in exc.value.args[0])
    with pytest.raises(AttributeError) as exc:
        lc.interact_bls(notebook_url=None)
    assert('object has no attribute' in exc.value.args[0])

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_graceful_exit_outside_notebook():
    """Test if running interact outside of a notebook does fails gracefully."""
    lcf = KeplerLightCurveFile(KEPLER10)
    lc = lcf.PDCSAP_FLUX.normalize().remove_nans().flatten()
    result = lc.interact_bls()
    assert(result is None)

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_helper_functions():
    """Can we use all the functions in interact_bls?"""
    from ..interact_bls import (prepare_bls_datasource,
                        prepare_folded_datasource, prepare_lightcurve_datasource)
    from ..interact_bls import (make_bls_figure_elements,
                                        make_folded_figure_elements,
                                        make_lightcurve_figure_elements)
    from ..interact_bls import (prepare_bls_help_source,
                                        prepare_f_help_source,
                                        prepare_lc_help_source)
    lcf = KeplerLightCurveFile(KEPLER10)
    lc = lcf.PDCSAP_FLUX.normalize().remove_nans().flatten()
    lc_source = prepare_lightcurve_datasource(lc)
    f_source = prepare_folded_datasource(lc.fold(1))
    model = BoxLeastSquares(lc.time, lc.flux)
    result = model.power([1,2,3], 0.3)
    bls_source = prepare_bls_datasource(result, 0)

    lc_help = prepare_lc_help_source(lc)
    f_help = prepare_f_help_source(lc.fold(1))
    bls_help = prepare_bls_help_source(bls_source, 1)

    make_lightcurve_figure_elements(lc, lc, lc_source, lc_source, lc_help)
    make_folded_figure_elements(lc.fold(1), lc.fold(1), f_source, f_source, f_help)
    make_bls_figure_elements(result, bls_source, bls_help)

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_full_widget():
    '''Test if we can run the widget with the keywords'''
    lcf = KeplerLightCurveFile(KEPLER10)
    lc = lcf.PDCSAP_FLUX.normalize().remove_nans().flatten()
    lc.interact_bls()
    lc.interact_bls(minimum_period=4)
    lc.interact_bls(maximum_period=5)
    lc.interact_bls(resolution=1000)

@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports,
                    reason="requires bokeh and astropy.stats.bls")
def test_tess_widget():
    '''Test if we can run the widget with the keywords'''
    lcf = TessLightCurveFile(TESS_SIM)
    lc = lcf.PDCSAP_FLUX.normalize().remove_nans().flatten()
    lc.interact_bls()
    lc.interact_bls(minimum_period=4)
    lc.interact_bls(maximum_period=5)
    lc.interact_bls(resolution=1000)
