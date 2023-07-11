"""Tests the features of the lightkurve.interact module."""
import warnings

from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from lightkurve import LightkurveWarning, LightkurveError
from lightkurve.search import search_targetpixelfile
from lightkurve.targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from .test_targetpixelfile import filename_tpf_tabby_lite
from lightkurve.interact import get_lightcurve_y_limits


bad_optional_imports = False
try:
    import bokeh
    from bokeh.plotting import ColumnDataSource
except ImportError:
    bad_optional_imports = True

example_tpf = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")
example_tpf_kepler = get_pkg_data_filename("data/test-tpf-kplr-tabby-first-cadence.fits")
example_tpf_tess = get_pkg_data_filename("data/tess25155310-s01-first-cadences.fits.gz")
example_tpf_tesscut = get_pkg_data_filename("data/test-tpf-tesscut_1x1.fits")
# Headers PMRA, PMDEC, PMTOTAL are removed
example_tpf_no_pm = get_pkg_data_filename("data/tess25155310-s01-first-cadences_no_pm.fits.gz")
# Headers for PM, ra/dec, and equinox all removed
example_tpf_no_target_position = get_pkg_data_filename("data/tess25155310-s01-first-cadences_no_target_position.fits.gz")


def test_bokeh_import_error(caplog):
    """If bokeh is not installed (optional dependency),
    is a friendly error message printed?"""
    try:
        import bokeh
    except ImportError:
        tpf = TessTargetPixelFile(example_tpf)
        tpf.interact()
        assert "requires the `bokeh` Python package" in caplog.text


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_malformed_notebook_url():
    """Test if malformed notebook_urls raise proper exceptions."""
    import bokeh

    tpf = TessTargetPixelFile(example_tpf)
    with pytest.raises(ValueError) as exc:
        tpf.interact(notebook_url="")
    assert "Empty host value" in exc.value.args[0]

@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_graceful_exit_outside_notebook():
    """Test if running interact outside of a notebook does fails gracefully."""
    import bokeh

    tpf = TessTargetPixelFile(example_tpf)
    result = tpf.interact()
    assert result is None


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_custom_aperture_mask():
    """Can we provide a custom lightcurve to show?"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_tabby_lite), TessTargetPixelFile(example_tpf)]
    import bokeh

    for tpf in tpfs:
        mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
        tpf.interact(aperture_mask=mask)
        mask = None
        tpf.interact(aperture_mask=mask)
        mask = "threshold"
        tpf.interact(aperture_mask=mask)


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_custom_exported_filename():
    """Can we provide a custom lightcurve to show?"""
    import bokeh

    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_tabby_lite), TessTargetPixelFile(example_tpf)]
    for tpf in tpfs:
        tpf.interact(exported_filename="demo.fits")
        tpf[0:2].interact()
        tpf[0:2].interact(exported_filename="string_only")
        tpf[0:2].interact(exported_filename="demo2.FITS")
        tpf[0:2].interact(exported_filename="demo3.png")
        tpf[0:2].interact(exported_filename="")
        tpf.interact(exported_filename=210690913)
        mask = tpf.time == tpf.time
        tpf[mask].interact()


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_transform_and_ylim_funcs():
    """Test the transform_func and ylim_func"""
    with warnings.catch_warnings():
        # Ignore the "TELESCOP is not equal to TESS" warning
        warnings.simplefilter("ignore", LightkurveWarning)
        tpfs = [KeplerTargetPixelFile(filename_tpf_tabby_lite), TessTargetPixelFile(example_tpf)]
    for tpf in tpfs:
        tpf.interact(transform_func=lambda lc: lc.normalize())
        tpf.interact(transform_func=lambda lc: lc.flatten().normalize())
        tpf.interact(transform_func=lambda lc: lc, ylim_func=lambda lc: (0, 2))
        tpf.interact(ylim_func=lambda lc: (0, lc.flux.max()))


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_functions():
    """Do the helper functions in the interact module run without syntax error?"""
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        prepare_lightcurve_datasource,
        aperture_mask_from_selected_indices,
        get_lightcurve_y_limits,
        make_lightcurve_figure_elements,
        make_tpf_figure_elements,
        show_interact_widget,
    )

    tpf = TessTargetPixelFile(example_tpf)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    # make the mask a bit more realistic
    mask[0, 0] = False
    mask[1, 2] = False

    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)

    # https://github.com/lightkurve/lightkurve/issues/990
    # ensure proper 2D - 1D conversion
    assert tpf_source.data["xx"].ndim == 1
    assert tpf_source.data["yy"].ndim == 1
    # for bokeh v3, .indices needs to plain list .
    # cf. https://github.com/bokeh/bokeh/issues/12624
    assert isinstance(tpf_source.selected.indices, list)

    # the lower-level function aperture_mask_from_selected_indices() is used in
    # callback _create_lightcurve_from_pixels(), which cannot be easily tested.
    # So we directly test it instead.
    assert_array_equal(aperture_mask_from_selected_indices(tpf_source.selected.indices, tpf), mask)

    lc = tpf.to_lightcurve(aperture_mask=mask)
    lc_source = prepare_lightcurve_datasource(lc)
    get_lightcurve_y_limits(lc_source)
    make_lightcurve_figure_elements(lc, lc_source)

    def ylim_func_sample(lc):
        return (np.nanpercentile(lc.flux, 0.1), np.nanpercentile(lc.flux, 99.9))

    make_lightcurve_figure_elements(lc, lc_source, ylim_func=ylim_func_sample)

    def ylim_func_unitless(lc):
        return (
            np.nanpercentile(lc.flux, 0.1).value,
            np.nanpercentile(lc.flux, 99.9).value,
        )

    make_lightcurve_figure_elements(lc, lc_source, ylim_func=ylim_func_unitless)

    make_tpf_figure_elements(tpf, tpf_source)
    show_interact_widget(tpf)


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
@pytest.mark.filterwarnings("ignore:Proper motion correction cannot be applied to the target")  # for TESSCut
@pytest.mark.parametrize("tpf_class, tpf_file, aperture_mask", [
    (TessTargetPixelFile, example_tpf_tess, "pipeline"),
    (TessTargetPixelFile, example_tpf_tesscut, "empty"),
    (KeplerTargetPixelFile, example_tpf_kepler, "threshold"),
    (TessTargetPixelFile, example_tpf_no_pm, "default"),
    ])
def test_interact_sky_functions(tpf_class, tpf_file, aperture_mask):
    """Do the helper functions in the interact module run without syntax error?"""
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_gaia_figure_elements,
    )
    tpf = tpf_class(tpf_file)
    mask = tpf._parse_aperture_mask(aperture_mask)
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig1, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_gaia_figure_elements(tpf, fig1)
    add_gaia_figure_elements(tpf, fig1, magnitude_limit=22)


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_case_no_target_coordinate():
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_gaia_figure_elements,
    )
    tpf_class, tpf_file = TessTargetPixelFile, example_tpf_no_target_position

    tpf = tpf_class(tpf_file)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig1, slider1 = make_tpf_figure_elements(tpf, tpf_source)
    with pytest.raises(LightkurveError, match=r".* no valid coordinate.*"):
        add_gaia_figure_elements(tpf, fig1)


@pytest.mark.remote_data
def test_interact_sky_functions_add_nearby_tics():
    """Test the backend of interact_sky() that combine Nearby TIC report with Gaia result."""
    from lightkurve.interact import (
        _get_nearby_gaia_objects,
        _add_nearby_tics_if_tess,
    )
    # This TIC's nearby report has a mix of stars with Gaia and without Gaia IDs.
    # https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=233087860
    tpf = search_targetpixelfile("TIC233087860", mission="TESS")[0].download()
    magnitude_limit = 17

    df_before = _get_nearby_gaia_objects(tpf, magnitude_limit)
    df, source_colnames_extras, tooltips_extras = _add_nearby_tics_if_tess(tpf, magnitude_limit, df_before)

    # based on what we know about the nearby report of the specific TIC,
    # some existing Gaia entries are added with tic data
    assert len(df[(df['Source'] > 0) & (df['tic'] != '')]) > 0

    # Some new entries with data only from TIC nearby report are added (hence no Gaia info)
    assert len(df[(df['Source'] == 0) & (df['tic'] != '')]) > 0


@pytest.mark.remote_data
def test_interact_sky_functions_add_nearby_tics_weird_dtype():
    """Test the backend of interact_sky() that combine Nearby TIC report with Gaia result.
    Case the dtype from Gaia result dataframe is weird.
    """
    from lightkurve.interact import (
        _get_nearby_gaia_objects,
        _add_nearby_tics_if_tess,
    )
    # For this TIC, the dataframe from Gaia search has weird DType:
    # df['Source'].dtype is an instance of pd.Int64Dtype, not the type class itself.
    # existing type check logic with np.issubdtype() fails with TypeError: Cannot interpret 'Int64Dtype()' as a data type
    tpf = search_targetpixelfile("TIC135100529", mission="TESS")[0].download()
    magnitude_limit = 18

    df_before = _get_nearby_gaia_objects(tpf, magnitude_limit)
    df, source_colnames_extras, tooltips_extras = _add_nearby_tics_if_tess(tpf, magnitude_limit, df_before)

    # some TICs are added successfully, without any error raised.
    assert len(df['tic'] != '') > 0


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_case_nearby_tics_failed(monkeypatch):
    """Test to ensure in case Nearby TIC service from ExoFOP not available,
       interact_sky will still function (without the TIC information) rather
       than raising exceptions.
    """
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_gaia_figure_elements,
    )
    import lightkurve.interact as lk_interact

    def mock_raise(*args):
        raise IOError("simulated service unavailable")

    monkeypatch.setattr(lk_interact, "_search_nearby_of_tess_target", mock_raise)

    tpf = TessTargetPixelFile(example_tpf_tess)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig1, slider1 = make_tpf_figure_elements(tpf, tpf_source)
    with pytest.warns(LightkurveWarning, match="cannot obtain nearby TICs"):
        add_gaia_figure_elements(tpf, fig1)


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_ylim_with_nans():
    """Regression test for #679: y limits should not be NaN."""
    lc_source = ColumnDataSource({"flux": [-1, np.nan, 1]})
    ymin, ymax = get_lightcurve_y_limits(lc_source)
    # ymin/ymax used to return nan, make sure this is no longer the case
    assert ymin == -1.176
    assert ymax == 1.176
