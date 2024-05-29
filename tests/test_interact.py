"""Tests the features of the lightkurve.interact module."""
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from lightkurve import LightkurveWarning, LightkurveError
from lightkurve.search import search_targetpixelfile
from lightkurve.targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from .test_targetpixelfile import filename_tpf_tabby_lite
from lightkurve.interact import get_lightcurve_y_limits
from lightkurve.interact_sky_providers import InteractSkyCatalogProvider, ProperMotionCorrectionMeta

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

#
# Tests for interact_sky()
#


class AbstractStubInteractSkyCatalogProvider(InteractSkyCatalogProvider):

    def query_catalog(self):
        if self.stub_data is None:
            return None
        tab = Table.read(self.stub_data, format="ascii")
        for c in tab.colnames:
            if c.lower() == "mag":
                tab[c].unit = u.mag
            elif c.lower() in ["pmra", "pmde"]:
                tab[c].unit = u.mas / u.year
            elif c.startswith("RA") or c.startswith("DE"):
                tab[c].unit = u.deg
        tab["magForSize"] = tab["Mag"]
        return tab

    def get_tooltips(self):
        return [
            ("ID", "@ID"),
            ("Mag", "@Mag"),
        ]

    def get_detail_view(self, data):
        return {
            "ID": data["ID"],
            "Mag": data["Mag"],
            }, None


class StubNoPMInteractSkyCatalogProvider(AbstractStubInteractSkyCatalogProvider):
    label = "stub_no_pm"

    stub_data = """\
    ID,RA,DEC,Mag
    1,30.00,45.00,11.0
    2,30.01,45.01,12.0
"""

    def get_proper_motion_correction_meta(self):
        return None


class StubWithPMInteractSkyCatalogProvider(AbstractStubInteractSkyCatalogProvider):
    label = "stub_with_pm"

    stub_data = """\
    ID,RAJ2000,DEJ2000,pmRA,pmDE,Mag
    1,30.00,45.00,1.1,1.2,11.0
    2,30.01,45.01,1.1,1.2,12.0
"""

    def get_proper_motion_correction_meta(self):
        J2000 = Time(2000.0, format="jyear", scale="tt")
        return ProperMotionCorrectionMeta("RAJ2000", "DEJ2000", "pmRA", "pmDE", J2000)

class StubEmptyResultInteractSkyCatalogProvider(StubWithPMInteractSkyCatalogProvider):
    label = "stub_empty_result"

    stub_data = """\
    ID,RAJ2000,DEJ2000,pmRA,pmDE,Mag
"""


class StubNoneResultInteractSkyCatalogProvider(StubWithPMInteractSkyCatalogProvider):
    label = "stub_empty_result"

    # some providers would return None for Empty result
    stub_data = None


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
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        init_provider,
        add_catalog_figure_elements
    )
    tpf = tpf_class(tpf_file)
    mask = tpf._parse_aperture_mask(aperture_mask)
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    for provider in [
        StubNoPMInteractSkyCatalogProvider(),
        StubWithPMInteractSkyCatalogProvider(),
        # test boundary cases
        StubEmptyResultInteractSkyCatalogProvider(),
        StubNoneResultInteractSkyCatalogProvider(),
    ]:
        init_provider(provider, tpf, magnitude_limit=18)
        renderer = add_catalog_figure_elements(provider, tpf, fig_tpf, message_selected_target, arrow_4_selected)
        assert renderer is not None


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_case_no_target_coordinate():
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        init_provider,
        add_catalog_figure_elements
    )
    tpf_class, tpf_file = TessTargetPixelFile, example_tpf_no_target_position

    tpf = tpf_class(tpf_file)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    with pytest.raises(LightkurveError, match=r".* no valid coordinate.*"):
        provider = StubWithPMInteractSkyCatalogProvider()
        init_provider(provider, tpf, magnitude_limit=18)
        add_catalog_figure_elements(provider, tpf, fig_tpf, message_selected_target, arrow_4_selected)


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_providers_sanity():
    """Basic sanity tests for supplied providers (ensure there is no syntax error, etc.)"""
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        init_provider,
        add_catalog_figure_elements
    )
    from lightkurve.interact_sky_providers import create_catalog_provider

    # known there are some data from all supported catalogs near this target
    tic, sector = 400621146, 71
    tpf = search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector, author="SPOC", exptime="short").download()
    mask = tpf._parse_aperture_mask("pipeline")
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    for provider_name in [
        "gaiadr3",
        "gaiadr3_tic",
        "vsx",
        "ztf",
    ]:
        provider = create_catalog_provider(provider_name)
        init_provider(provider, tpf, magnitude_limit=18)
        renderer = add_catalog_figure_elements(provider, tpf, fig_tpf, message_selected_target, arrow_4_selected)
        assert renderer is not None


@pytest.mark.remote_data
def test_interact_sky_provider_gaiadr3_tic():
    """Test Gaia DR3 + TIC join"""
    from lightkurve.interact_sky_providers import create_catalog_provider

    #
    # Test 1: TIC 233087860
    # Is nearby results has a mix of TIC with and without Gaia cross-match
    #   https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=233087860
    #   ^^^ the page is for Gaia DR2, but Gaia DR3 result is similar.
    ra, dec = 272.20452, 60.678785

    tpf_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    provider = create_catalog_provider("gaiadr3_tic")
    provider.init(coord=tpf_coord, radius=75*u.arcsec, magnitude_limit=18)
    rs = provider.query_catalog()

    # print(rs)  # for debugging
    # known there are multiple rows that have both Gaia DR3 Source and TIC (cross-matched)
    assert len(rs[(rs["Source"] != "") & (rs["TIC"] != "")]) > 1
    # known there are rows that have nave  no Gaia DR3 Source (TIC without Gaia)
    assert len(rs[(rs["Source"] == "") & (rs["TIC"] != "")]) > 0
    # Expected cross-match of the target
    assert rs[rs["TIC"] == "233087860"]["Source"][0] == "2158781336134901760"

    #
    # Tests 2 and 3: TIC 167092385
    # - the TIC has not Gaia DR2 Source (in TIC v8.2),
    #   so the actual correspond Gaia DR3 entry appears as a separate row
    # - also test duplicates removal:  nearby TIC 167092380 is split
    ra, dec = 318.72463517654, 38.09423368095

    tpf_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    provider = create_catalog_provider("gaiadr3_tic")
    provider.init(coord=tpf_coord, radius=15*u.arcsec, magnitude_limit=18)
    rs = provider.query_catalog()

    # Expected cross-match failure
    assert rs[rs["TIC"] == "167092385"]["Source"][0] == ""
    # the actual correspond Gaia DR3 entry
    assert rs[rs["Source"] == "1964797660741411072"]["TIC"][0] == ""

    # Ensure it does not return duplicates by default
    assert "167092380" in rs["TIC"], "the splitted TIC should be present"
    assert "1961258180" in rs["TIC"], "one of the split entry, that should also be present"
    assert "1961258171" not in rs["TIC"], "the other split entry, marked as duplicate, and should not show up"

    #
    # Test 4: TIC 440100539
    # - exclude ARTIFACTs (from 2MASS) by default
    ra, dec = 4.35804798287, 42.28232330179

    tpf_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    provider = create_catalog_provider("gaiadr3_tic")
    provider.init(coord=tpf_coord, radius=10*u.arcsec, magnitude_limit=18)
    rs = provider.query_catalog()

    # Expected the target is found
    assert len(rs[rs["TIC"] == "440100539"]) > 0, "The target should be present"
    assert len(rs[rs["TIC"] == "440100538"]) == 0, "The nearby 2MASS artifact should be excluded"


# TODO: test VSX parsing edge cases


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_ylim_with_nans():
    """Regression test for #679: y limits should not be NaN."""
    lc_source = ColumnDataSource({"flux": [-1, np.nan, 1]})
    ymin, ymax = get_lightcurve_y_limits(lc_source)
    # ymin/ymax used to return nan, make sure this is no longer the case
    assert ymin == -1.176
    assert ymax == 1.176
