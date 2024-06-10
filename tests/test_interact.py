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
    stub_data = """\
    ID,RA,DEC,Mag
    1,30.00,45.00,11.0
    2,30.01,45.01,12.0
"""

    @property
    def label(self):
        return "stub_no_pm"

    def get_proper_motion_correction_meta(self):
        return None


class StubWithPMInteractSkyCatalogProvider(AbstractStubInteractSkyCatalogProvider):
    stub_data = """\
    ID,RAJ2000,DEJ2000,pmRA,pmDE,Mag
    1,30.00,45.00,1.1,1.2,11.0
    2,30.01,45.01,1.1,1.2,12.0
"""


    @property
    def label(self):
        return "stub_with_pm"

    def get_proper_motion_correction_meta(self):
        J2000 = Time(2000.0, format="jyear", scale="tt")
        return ProperMotionCorrectionMeta("RAJ2000", "DEJ2000", "pmRA", "pmDE", "icrs", J2000)


class StubWithPMnFK5CoordInteractSkyCatalogProvider(StubWithPMInteractSkyCatalogProvider):
    @property
    def label(self):
        return "stub_with_pm_and_fk5_coord"

    def get_proper_motion_correction_meta(self):
        J2000 = Time(2000.0, format="jyear", scale="tt")
        return ProperMotionCorrectionMeta("RAJ2000", "DEJ2000", "pmRA", "pmDE", "fk5", J2000)


class StubEmptyResultInteractSkyCatalogProvider(StubWithPMInteractSkyCatalogProvider):
    stub_data = """\
    ID,RAJ2000,DEJ2000,pmRA,pmDE,Mag
"""

    @property
    def label(self):
        return "stub_empty_result"


class StubNoneResultInteractSkyCatalogProvider(StubWithPMInteractSkyCatalogProvider):
    # some providers would return None for Empty result
    stub_data = None

    @property
    def label(self):
        return "stub_none_result"


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
        parse_and_add_catalogs_figure_elements
    )

    tpf = tpf_class(tpf_file)
    mask = tpf._parse_aperture_mask(aperture_mask)
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    catalogs = [
        StubNoPMInteractSkyCatalogProvider,
        StubWithPMInteractSkyCatalogProvider,
        StubWithPMnFK5CoordInteractSkyCatalogProvider,  # catalog coordinate in FK5 rather than ICRS
        # test boundary cases
        StubEmptyResultInteractSkyCatalogProvider,
        StubNoneResultInteractSkyCatalogProvider,
    ]
    magnitude_limit = 18
    providers, renderers = parse_and_add_catalogs_figure_elements(
        catalogs, magnitude_limit, tpf, fig_tpf, message_selected_target, arrow_4_selected
        )
    for _, _, renderer in zip(catalogs, providers, renderers):  # zip to ensure they are all of the same length
        assert renderer is not None


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_case_no_target_coordinate():
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        parse_and_add_catalogs_figure_elements
    )
    tpf_class, tpf_file = TessTargetPixelFile, example_tpf_no_target_position

    tpf = tpf_class(tpf_file)
    mask = tpf.flux[0, :, :] == tpf.flux[0, :, :]
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    with pytest.raises(LightkurveError, match=r".* no valid coordinate.*"):
        catalogs = [StubWithPMInteractSkyCatalogProvider]
        magnitude_limit = 18
        parse_and_add_catalogs_figure_elements(
            catalogs, magnitude_limit, tpf, fig_tpf, message_selected_target, arrow_4_selected
        )


@pytest.mark.remote_data
@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_interact_sky_functions_providers_sanity():
    """Basic sanity tests for supplied providers (ensure there is no syntax error, etc.)
    It also test the parsing of catalogs parameter.
    """
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        parse_and_add_catalogs_figure_elements,
        _row_to_dict
    )

    # known there are some data from all supported catalogs near this target
    tic, sector = 400621146, 71
    tpf = search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector, author="SPOC", exptime="short").download()
    mask = tpf._parse_aperture_mask("pipeline")
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    catalogs = [
        "gaiadr3",
        "gaiadr3_tic",
        # case a catalog is defined with override parameter
        ("ztf", dict(radius=6*u.arcsec)),
        # the commented override parameter part of the test,
        # simulating case an user comments it out, leaving a tuple
        ("vsx",
            #  dict(radius=4*u.arsec),
        ),
    ]
    magnitude_limit = 18
    providers, renderers = parse_and_add_catalogs_figure_elements(
        catalogs, magnitude_limit, tpf, fig_tpf, message_selected_target, arrow_4_selected
        )
    for _, provider, renderer in zip(catalogs, providers, renderers):  # zip to ensure they are all of the same length
        assert renderer is not None
        # For the purpose of the sanity test, each provider should have at least 1 record to exercise typical code path
        # The coordinate / radius is selected to be so, and should be stable.
        assert len(renderer.data_source.data["ra"]) > 0, f"Provider {provider.label} should have at least 1 record."

        assert len(provider.get_tooltips()) > 0,  "get_tooltips() should not cause error, and return a non-empty list"
        # simulate getting the detail view of the target (tap it)
        details, extra_rows = provider.get_detail_view(_row_to_dict(renderer.data_source, 0))
        assert len(details) > 0,  "get_detail_view() should not cause error, and return a non-empty dict"


@pytest.mark.remote_data
def test_interact_sky_provider_gaiadr3_detail_view():
    """Sanity test for Gaia DR3 detail view, which has some extra logic, e.g., NSS flag parsing."""
    import bokeh
    from lightkurve.interact import (
        prepare_tpf_datasource,
        make_tpf_figure_elements,
        add_target_figure_elements,
        make_interact_sky_selection_elements,
        parse_and_add_catalogs_figure_elements,
        _row_to_dict
    )

    # known target with both VARIABLE flag and NSS flag on, which would
    # require the special logic in get_detail_view()
    tic, sector = 229647506, 73
    tpf = search_targetpixelfile(f"TIC{tic}", mission="TESS", sector=sector, author="SPOC", exptime="short").download()
    mask = tpf._parse_aperture_mask("pipeline")
    tpf_source = prepare_tpf_datasource(tpf, aperture_mask=mask)
    fig_tpf, slider1 = make_tpf_figure_elements(tpf, tpf_source, tpf_source_selectable=False)
    add_target_figure_elements(tpf, fig_tpf)
    message_selected_target, arrow_4_selected = make_interact_sky_selection_elements(fig_tpf)

    catalogs = [
        ("gaiadr3", dict(radius=5*u.arcsec)),
    ]
    magnitude_limit = 18

    provider, renderer = parse_and_add_catalogs_figure_elements(
        catalogs, magnitude_limit, tpf, fig_tpf, message_selected_target, arrow_4_selected
        )
    provider, renderer = provider[0], renderer[0]  # only 1 catalog
    details, extra_rows = provider.get_detail_view(_row_to_dict(renderer.data_source, 0))
    # for now, just to ensure the call do not cause errors, so no extra assertion


@pytest.mark.remote_data
def test_interact_sky_provider_gaiadr3_tic():
    """Test Gaia DR3 + TIC join"""
    from lightkurve.interact_sky_providers import resolve_catalog_provider_class

    #
    # Test 1: TIC 233087860
    # Is nearby results has a mix of TIC with and without Gaia cross-match
    #   https://exofop.ipac.caltech.edu/tess/nearbytarget.php?id=233087860
    #   ^^^ the page is for Gaia DR2, but Gaia DR3 result is similar.
    ra, dec = 272.20452, 60.678785

    tpf_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    provider = resolve_catalog_provider_class("gaiadr3_tic")(coord=tpf_coord, radius=75*u.arcsec, magnitude_limit=18)
    rs = provider.query_catalog()

    # print(rs)  # for debugging
    # known there are multiple rows that have both Gaia DR3 Source and TIC (cross-matched)
    assert len(rs[(rs["Source"] != "") & (rs["TIC"] != "")]) > 1
    # known there are rows that have nave  no Gaia DR3 Source (TIC without Gaia)
    assert len(rs[(rs["Source"] == "") & (rs["TIC"] != "")]) > 0
    # Expected cross-match of the target
    assert rs[rs["TIC"] == "233087860"]["Source"][0] == "2158781336134901760"

    # Test 1a,for rows with no Gaia data, ensure correct `fill_value` is used for missing values
    # (bokeh data source does not support missing value)
    nss_filled = rs[(rs["Source"] == "") & (rs["TIC"] != "")]["NSS"].filled()
    assert_array_equal(nss_filled,  np.full_like(nss_filled, 0))

    #
    # Tests 2 and 3: TIC 167092385
    # - the TIC has not Gaia DR2 Source (in TIC v8.2),
    #   so the actual correspond Gaia DR3 entry appears as a separate row
    # - also test duplicates removal:  nearby TIC 167092380 is split
    ra, dec = 318.72463517654, 38.09423368095

    tpf_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
    provider = resolve_catalog_provider_class("gaiadr3_tic")(coord=tpf_coord, radius=15*u.arcsec, magnitude_limit=18)
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
    provider = resolve_catalog_provider_class("gaiadr3_tic")(coord=tpf_coord, radius=10*u.arcsec, magnitude_limit=18)
    rs = provider.query_catalog()

    # Expected the target is found
    assert len(rs[rs["TIC"] == "440100539"]) > 0, "The target should be present"
    assert len(rs[rs["TIC"] == "440100538"]) == 0, "The nearby 2MASS artifact should be excluded"


@pytest.mark.remote_data
def test_interact_sky_provider_vsx():
    """Test VSX"""
    from lightkurve.interact_sky_providers import resolve_catalog_provider_class

    # case the coordinate has PM
    tpf_coord = SkyCoord(
        196.421 * u.deg, 18.01 * u.deg, frame="icrs",
        pm_ra_cosdec=1.0 * u.milliarcsecond / u.year, pm_dec=1.0 * u.milliarcsecond / u.year, obstime=Time(1234, format="btjd")
    )
    provider = resolve_catalog_provider_class("vsx")(coord=tpf_coord, radius=60*u.arcsec, magnitude_limit=20)
    rs = provider.query_catalog()
    assert len(rs) > 0

    # case the coordinate has no PM
    tpf_coord = SkyCoord(
        196.421 * u.deg, 18.01 * u.deg, frame="icrs",
    )
    provider = resolve_catalog_provider_class("vsx")(coord=tpf_coord, radius=60*u.arcsec, magnitude_limit=20)
    rs = provider.query_catalog()
    assert len(rs) > 0


VSX_RESPONSE_TEST_CASES = dict(
    # implementation-specific tests:
    # astropy table by default would interpret columns Period as str (general case, e.g., values with uncertain flag ":"), or float (when the data are all numbers)
    #
    # https://www.aavso.org/vsx/index.php?view=api.list&ra=283.2372404664148&dec=58.21640948105657&radius=0.06416666666666666&format=json
    has_epoch_period_as_float_1row="""\
{"VSXObjects":{"VSXObject":[{"Name":"ASASSN-V J185256.94+581259.1","RA2000":"283.23724","Declination2000":"58.21641","VariabilityType":"EA","Period":"5.09843","Epoch":"2457991.86","MaxMag":"14.21 V","MinMag":"14.37 V","Category":"Variable","OID":"1526644","Constellation":"Dra"}]}}
""",

    # handle case a column (Epoch) is missing in JSON
    # https://www.aavso.org/vsx/index.php?view=api.list&ra=316.12157123252234&dec=70.76451425946803&radius=0.06416666666666666&format=json
    no_epoch_as_float_1row="""\
{"VSXObjects":{"VSXObject":[{"Name":"ASAS J210430+7045.9","RA2000":"316.12069","Declination2000":"70.76411","ProperMotionRA":"96.7230","ProperMotionDec":"13.3200","VariabilityType":"EW","Period":"0.306243","MaxMag":"8.794 V","MinMag":"(0.20) V","SpectralType":"K2","Category":"Variable","OID":"300626","Constellation":"Cep"}]}}
""",

    # https://www.aavso.org/vsx/index.php?view=api.list&ra=285.31165553284677&dec=60.87812797101079&radius=0.07583333333333334&format=json
    empty_result="""\
{"VSXObjects":[]}
""",

    # Average case with multiple rows, which also tests various edge cases, including
    # - uncertain period: Y Com
    # - no epoch: GP Com
    # - limit flags on Max/MinFlag: Z Com [*]
    # - MinMag is amplitude: ASAS J130830+1828.1
    # - MinFlag amplitude has a different passband: CSS_J130149.6+181629 [*]
    # - MinMag is "?": LINEAR 8078628 [*]
    # - minimum info (no PM, variability type, magnitude range/amplitude, etc.) : CSS_J130246.9+182141 [*]
    # [*] the JSON response is modified (from actual VSX response) to test edge case
    # https://www.aavso.org/vsx/index.php?view=api.list&ra=196.421&dec=18.01&radius=2.1&format=json
    avg_case="""\
{"VSXObjects":{"VSXObject":[
{"Name": "Y Com", "AUID": "000-BDM-173", "RA2000": "196.64292", "Declination2000": "18.97247", "ProperMotionRA": "-6.2000", "ProperMotionDec": "-3.4000", "VariabilityType": "SRB", "Period": "95:", "Epoch": "2452994", "MaxMag": "13.4 B", "MinMag": "14.39 B", "Category": "Variable", "OID": "9610", "Constellation": "Com"},
{"Name": "Z Com", "AUID": "000-BDM-174", "RA2000": "197.07607", "Declination2000": "18.54055", "ProperMotionRA": "4.5000", "ProperMotionDec": "-22.3000", "VariabilityType": "RRAB", "Period": "0.5466892", "Epoch": "2453469.8819", "RiseDuration": "13", "MaxMag": "<13.14 V", "MinMag": ">14.29 V", "Category": "Variable", "OID": "9611", "Constellation": "Com"},
{"Name": "GP Com", "AUID": "000-BCV-502", "RA2000": "196.42667", "Declination2000": "18.01772", "ProperMotionRA": "-343.0000", "ProperMotionDec": "37.0000", "VariabilityType": "IBWD", "Period": "0.032339", "MaxMag": "15.7 V", "MinMag": "16.2 V", "SpectralType": "DBe", "Category": "Variable", "OID": "9800", "Constellation": "Com"},
{"Name": "ASAS J130830+1828.1", "RA2000": "197.12658", "Declination2000": "18.47531", "ProperMotionRA": "-20.1000", "ProperMotionDec": "-2.7000", "VariabilityType": "ED", "Period": "0.448945", "MaxMag": "12.382 V", "MinMag": "(0.160) V", "Category": "Variable", "OID": "281437", "Constellation": "Com"},
{"Name": "LINEAR 8078628", "RA2000": "195.73021", "Declination2000": "17.55149", "ProperMotionRA": "-32.5000", "ProperMotionDec": "-9.7000", "VariabilityType": "EW", "Period": "0.940526", "MaxMag": "15.51 CV", "MinMag": "?", "Category": "Variable", "OID": "319884", "Constellation": "Com"},
{"Name": "CSS_J130149.6+181629", "RA2000": "195.45693", "Declination2000": "18.27497", "ProperMotionRA": "-2.4000", "ProperMotionDec": "-5.3000", "VariabilityType": "RRAB", "Period": "0.6422550", "Epoch": "2453470.3198", "MaxMag": "16.773 CV", "MinMag": "(0.85) TESS", "Category": "Variable", "OID": "291920", "Constellation": "Com"},
{"Name": "CSS_J130246.9+182141", "RA2000": "195.69565", "Declination2000": "18.36147", "MaxMag": "18.399 CV", "Category": "Suspected", "OID": "291928", "Constellation": "Com"},
{"Name": "CSS_J130256.1+182529", "RA2000": "195.73391", "Declination2000": "18.42494", "VariabilityType": "RRAB", "Period": "0.5606905", "Epoch": "2453470.1252", "MaxMag": "17.903 CV", "MinMag": "(0.54) CV", "Category": "Variable", "OID": "291929", "Constellation": "Com"},
{"Name": "CSS_J130323.8+183610", "RA2000": "195.84935", "Declination2000": "18.60292", "VariabilityType": "RRAB", "Period": "0.6153327", "Epoch": "2453469.8706", "MaxMag": "18.745 CV", "MinMag": "(0.59) CV", "Category": "Variable", "OID": "291935", "Constellation": "Com"},
{"Name": "CSS_J130450.2+183549", "RA2000": "196.20921", "Declination2000": "18.59704", "ProperMotionRA": "2.3000", "ProperMotionDec": "0.5000", "VariabilityType": "RRAB", "Period": "0.71316", "Epoch": "2455323.7766", "MaxMag": "16.943 CV", "MinMag": "(0.75) CV", "Category": "Variable", "OID": "291955", "Constellation": "Com"},
{"Name": "CSS_J130655.9+180243", "RA2000": "196.73313", "Declination2000": "18.04544", "ProperMotionRA": "-0.8000", "ProperMotionDec": "-5.9000", "VariabilityType": "RRAB", "Period": "0.65192", "Epoch": "2455664.7858", "MaxMag": "16.654 CV", "MinMag": "(0.73) CV", "Category": "Variable", "OID": "291984", "Constellation": "Com"},
{"Name": "CSS_J130829.7+175900", "RA2000": "197.12409", "Declination2000": "17.98351", "VariabilityType": "RRAB", "Period": "0.5290068", "Epoch": "2453470.3275", "MaxMag": "18.679 CV", "MinMag": "(1.07) CV", "Category": "Variable", "OID": "292001", "Constellation": "Com"},
{"Name": "V0337 Com", "RA2000": "195.55312", "Declination2000": "17.83928", "VariabilityType": "RRD", "Period": "0.355745", "Epoch": "2455000.342", "MaxMag": "18.1 CV", "MinMag": "18.9 CV", "Category": "Variable", "OID": "382137", "Constellation": "Com"},
{"Name": "V0338 Com", "RA2000": "195.93631", "Declination2000": "17.96857", "VariabilityType": "RRD", "Period": "0.361005", "Epoch": "2455000.295", "MaxMag": "18.2 CV", "MinMag": "19.2 CV", "Category": "Variable", "OID": "382159", "Constellation": "Com"},
{"Name": "CSS_J130600.1+180046", "AUID": "000-BPF-391", "RA2000": "196.50058", "Declination2000": "18.01297", "VariabilityType": "EB", "Period": "1.30748", "MaxMag": "16.68 CV", "MinMag": "(0.14) CV", "Category": "Variable", "OID": "382191", "Constellation": "Com"},
{"Name": "CSS_J130915.4+181943", "RA2000": "197.31458", "Declination2000": "18.32863", "ProperMotionRA": "11.1000", "ProperMotionDec": "-8.3000", "VariabilityType": "EA", "Period": "0.82638", "Epoch": "2458526.9851", "MaxMag": "15.951 r", "MinMag": "(0.284) r", "Category": "Variable", "OID": "382234", "Constellation": "Com"},
{"Name": "ZTF J130403.26+184147.6", "RA2000": "196.01360", "Declination2000": "18.69656", "VariabilityType": "BY", "Period": "1.2442269", "Epoch": "2458487.0296", "MaxMag": "16.900 r", "MinMag": "(0.099) r", "Discoverer": "Zwicky Transient Facility (ZTF)", "Category": "Variable", "OID": "1704066", "Constellation": "Com"},
{"Name": "ZTF J130429.63+180459.3", "RA2000": "196.12348", "Declination2000": "18.08316", "VariabilityType": "BY", "Period": "0.5901588", "Epoch": "2458295.7337", "MaxMag": "16.524 r", "MinMag": "(0.106) r", "Discoverer": "Zwicky Transient Facility (ZTF)", "Category": "Variable", "OID": "1704070", "Constellation": "Com"},
{"Name": "ZTF J130917.23+175650.1", "RA2000": "197.32183", "Declination2000": "17.94727", "VariabilityType": "RS", "Period": "0.2722320", "Epoch": "2458210.8244", "MaxMag": "13.397 g", "MinMag": "(0.117) g", "Discoverer": "Zwicky Transient Facility (ZTF)", "Category": "Variable", "OID": "1750641", "Constellation": "Com"},
{"Name": "ASASSN-V J130329.61+184723.0", "RA2000": "195.87340", "Declination2000": "18.78974", "VariabilityType": "ROT", "Period": "2.9103", "Epoch": "2458933.393", "MaxMag": "15.07 g", "MinMag": "(0.34) g", "Category": "Variable", "OID": "2272914", "Constellation": "Com"},
{"Name": "ASASSN-V J130647.48+182752.4", "RA2000": "196.69797", "Declination2000": "18.46471", "VariabilityType": "ROT", "Period": "16.919", "Epoch": "2459363.613", "MaxMag": "13.16 g", "MinMag": "(0.13) g", "Category": "Variable", "OID": "2273094", "Constellation": "Com"},
{"Name": "ASASSN-V J130651.61+174203.2", "RA2000": "196.71500", "Declination2000": "17.70097", "VariabilityType": "ROT", "Period": "12.311", "Epoch": "2458923.442", "MaxMag": "15.26 g", "MinMag": "(0.32) g", "Category": "Variable", "OID": "2273100", "Constellation": "Com"}
]}}
"""
)


@pytest.mark.parametrize("case_name", list(VSX_RESPONSE_TEST_CASES.keys()))
def test_interact_sky_vsx_parse_json(case_name):
    """Test various edge cases in parsing VSX JSON result as astropy table"""
    import json
    from lightkurve.interact_sky_providers.vsx import _parse_response

    json_obj = json.loads(VSX_RESPONSE_TEST_CASES.get(case_name))
    tab = _parse_response(json_obj)
    # if tab is not None:
    #     tab.pprint_all()  # for debug purpose

    # just test the parsing does not cause errors, and return
    # None or a table
    assert tab is None or len(tab) >= 0


@pytest.mark.skipif(bad_optional_imports, reason="requires bokeh")
def test_ylim_with_nans():
    """Regression test for #679: y limits should not be NaN."""
    lc_source = ColumnDataSource({"flux": [-1, np.nan, 1]})
    ymin, ymax = get_lightcurve_y_limits(lc_source)
    # ymin/ymax used to return nan, make sure this is no longer the case
    assert ymin == -1.176
    assert ymax == 1.176
