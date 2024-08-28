from typing import Tuple, Union
import warnings

import numpy as np

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table, MaskedColumn, join
from astropy.time import Time

import astroquery.simbad as simbad
import astroquery.vizier as vizier

from .core import ProperMotionCorrectionMeta

from .vizier import VizierInteractSkyCatalogProvider, _query_cone_region


def _decode_gaiadr3_nss_flag(nss_flag):
    """Decode NSS (NON_SINGLE_STAR) flag in Gaia DR3 Main.
    Reference:
    https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html#p344
    """
    flags = []
    for mask, nss_type in [
        (0b1, "AB"),  # astrometric binary
        (0b10, "SB"),  # spectroscopic binary
        (0b100, "EB"),  # eclipsing binary
    ]:
        if nss_flag & mask > 0:
            flags.append(nss_type)
    return flags


def _fill_template(template, var_value, var_name="%s"):
    return template.replace(var_name, str(var_value))


class GaiaDR3InteractSkyCatalogProvider(VizierInteractSkyCatalogProvider):
    """
    Provide Gaia DR3 data to `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    Gaia DR3 data is from Vizier: https://cdsarc.cds.unistra.fr/viz-bin/cat/I/355

    The class is used by ``interact_sky()`` internally. The behavior can
    be customized by supplying a dictionary of keyword parameters to
    `catalogs` parameter of ``interact_sky()``. The keyword parameters are
    then used to customize the parameters passed to the constructor here.

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        the coordinate of the target.

    radius: float or `~astropy.units.Quantity`
        the cone search radius, in arc seconds if the value is float.

    magnitude_limit: float
        A value to limit the results in based on Gaia Gmag.

    scatter_kwargs: dict
        keyword arguments passed to bokeh's ``figure.scatter()``
        function to plot the stars.

    extra_cols_in_detail_view: dict, optional
        additional Gaia DR3 parameters to be included in detail view, in the form
        of (column name in Gaia DR3 Vizier table, display name).

    url_templates: dict, optional
        The URL templates for the URL shown in detail view. The keys are:

        * ``gaiadr3_main_url`` : Gaia DR3 Main
        * ``gaiadr3_var_url`` : Gaia DR3 Variable
        * ``gaiadr3_nss_url`` : Gaia DR3 Non Single Star
        * ``simbad_url_by_gaia_source`` : SIMBAD by Gaia Source
        * ``simbad_url_by_coord`` : SIMBAD by coordinate

        For all except ``simbad_url_by_coord``, the string ``%s`` is replaced by
        Gaia source. For ``simbad_url_by_coord``, the strings ``%ra`` and ``%dec``
        are replaced by the target's right ascension and declination
        (in degrees) respectively.
    """

    # Gaia DR3 reference epoch: 2016.0,  time coordinate: barycentric coordinate time (TCB).
    # https://www.cosmos.esa.int/web/gaia/dr3
    # J2016 = Time(2016.0, format="jyear", scale="tcb")

    # OPEN: unsure if the epoch RAJ200/DEJ200 is in tt or possibly tdb, etc.
    J2000 = Time(2000.0, format="jyear", scale="tt")

    @property
    def url_templates_defaults(self):
        """The default URL templates for links to Gaia / SIMBAD data.
        For most templates, the string `%s` will be replaced by Gaia Source value.
        The one exception is `simbad_url_by_coord`. `%ra` and `%dec` will be replaced
        by the target's RA / DEC values in degrees.
        """
        # reuse the Vizier / SIMBAD servers specified in astroquery
        vizier_server = vizier.conf.server
        simbad_server = simbad.conf.server
        return dict(
            gaiadr3_main_url=f"https://{vizier_server}/viz-bin/VizieR-4?-source=+I%2F355%2Fgaiadr3+I%2F355%2Fparamp&Source=%s",
            gaiadr3_var_url=f"https://{vizier_server}/viz-bin/VizieR-4?-source=+I%2F358%2Fvarisum+I%2F358%2Fvclassre&Source=%s",
            gaiadr3_nss_url=f"https://{vizier_server}/viz-bin/VizieR-4?-ref=VIZ65a1a2351812e4&-source=I%2F357&Source=%s",
            simbad_url_by_gaia_source=f"https://{simbad_server}/simbad/sim-id?Ident=Gaia DR3 %s",
            # the by_coord template is special with 2 variables, %ra and %dec
            simbad_url_by_coord=f"https://{simbad_server}/simbad/sim-coo?Coord=%ra+%dec&Radius=2&Radius.unit=arcmin",
        )

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
        extra_cols_in_detail_view: dict = None,
        url_templates: dict = None,
    ) -> None:
        if scatter_kwargs is None:
            scatter_kwargs = dict(
                marker="circle",
                fill_alpha=0.3,
                line_color=None,
                selection_color="firebrick",
                nonselection_fill_alpha=0.3,
                nonselection_line_color=None,
                nonselection_line_alpha=1.0,
                fill_color="firebrick",
                hover_fill_color="firebrick",
                hover_alpha=0.9,
                hover_line_color="white",
            )
        super().__init__(coord, radius, magnitude_limit, scatter_kwargs)
        # Gaia DR3 Vizier specific
        self.catalog_name = "I/355/gaiadr3"
        self.magnitude_limit_column_name = "Gmag"
        self.cols_for_source = [
            "Source",
            "Gmag",
            "Plx",
            "VarFlag",
            "NSS",
        ]
        self.columns = self.cols_for_source + ["RAJ2000", "DEJ2000", "pmRA", "pmDE"]
        # Gaia columns that could have large integers
        self.cols_as_str_for_source = ["Source", "SolID"]
        if extra_cols_in_detail_view is not None:
            self.extra_cols_in_detail_view = extra_cols_in_detail_view
            cols = extra_cols_in_detail_view.keys()
            # include them in the query, and the data source
            self.columns += cols
            self.cols_for_source += cols
        else:
            self.extra_cols_in_detail_view = None
        self.url_templates = self.url_templates_defaults
        if url_templates is not None:
            self.url_templates.update(url_templates)

    @property
    def label(self) -> str:
        return "Gaia DR3"

    def query_catalog(self) -> Table:
        tab = super().query_catalog()
        self._post_process_gaiadr3_result(tab)
        return tab

    def _post_process_gaiadr3_result(self, tab):
        # set custom fill_value for some columns, typically integer columns,
        # so that for rows with missing values, the custom `fill_value` is used
        # for column NSS, without setting fill_value, the astropy default `fill_value`
        # is often 63, confusing  users.
        if "NSS" in tab.colnames:
            tab["NSS"].fill_value = 0

    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        # Use RAJ200/ DEJ2000 instead of Gaia DR3's native RA_IRCS in J2016.0 for ease of
        # merging with the result from TIC (which also has J2000 coordinate)
        # If more precise correction is needed for Gaia + TIC,
        # we need to do the correction on case-by-case basis
        # - Gaia DR3: use RA_ICRS in J2016.0
        # - TICs without Gaia DR3 match: use RA_orig,
        #   the origin of the coordinate is denoted in POSflag (Gaia DR2 / 1, 2MASS, hip, etc.)
        #   need to find the reference epoch of each of the origin catalog, e.g. J2015.5 for Gaia DR2
        return ProperMotionCorrectionMeta(
            "RAJ2000", "DEJ2000", "pmRA", "pmDE", "icrs", self.J2000
        )

    def add_to_data_source(self, result: Table, source: dict) -> None:
        super().add_to_data_source(result, source)
        source["one_over_plx"] = 1.0 / (result["Plx"] / 1000.0)

    def get_tooltips(self) -> list:
        return [
            ("Gaia Source", "@Source"),
            ('Separation (")', "@separation{0.00}"),
            ("Gmag", "@Gmag"),
            ("Parallax (mas)", "@Plx{0.000} (~@one_over_plx{0,0} pc)"),
            ("RA", "@ra{0,0.00000000}"),
            ("DEC", "@dec{0,0.00000000}"),
            ("column", "@x{0.0}"),
            ("row", "@y{0.0}"),
            ("Variable", "@VarFlag"),
            ("NSS", "@NSS"),
        ]

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        # the vizier URL returns both Gaia DR3 Main and Gaia DR3 Astrophysical parameters table for convenience
        gaiadr3_main_url = _fill_template(
            self.url_templates["gaiadr3_main_url"], data["Source"]
        )
        simbad_url_by_gaia_source = _fill_template(
            self.url_templates["simbad_url_by_gaia_source"], data["Source"]
        )
        simbad_url_by_coord = _fill_template(
            self.url_templates["simbad_url_by_coord"], data["ra"], var_name="%ra"
        )
        simbad_url_by_coord = _fill_template(
            simbad_url_by_coord, data["dec"], var_name="%dec"
        )
        if data["Source"] != "":
            source_val_html = f"""{data['Source']} (<a href="{gaiadr3_main_url}" target="_blank">Vizier</a>)"""
            extra_rows = [
                f'<a target="_blank" href="{simbad_url_by_gaia_source}">SIMBAD by Gaia Source</a>',
                f'<a target="_blank" href="{simbad_url_by_coord}">SIMBAD by coordinate</a>',
            ]
        else:
            source_val_html = ""
            extra_rows = []

        var_html = data["VarFlag"]
        if var_html == "VARIABLE":
            gaiadr3_var_url = _fill_template(
                self.url_templates["gaiadr3_var_url"], data["Source"]
            )
            var_html += f' (<a href="{gaiadr3_var_url}" target="_blank">Vizier</a>)'

        nss_html = str(data["NSS"])
        if data["NSS"] != 0:
            flags_text = ", ".join(_decode_gaiadr3_nss_flag(data["NSS"]))
            gaiadr3_nss_url = _fill_template(
                self.url_templates["gaiadr3_nss_url"], data["Source"]
            )
            nss_html += f' ({flags_text})&emsp;(<a href="{gaiadr3_nss_url}" target="_blank">Vizier</a>)'

        key_vals = {
            "Source": source_val_html,
            'Separation (")': f"{data['separation']:.2f}",
            "Gmag": f"{data['Gmag']:.3f}",
            "Parallax (mas)": f"{data['Plx']:.3f} (~ {data['one_over_plx']:,.0f} pc)",
            "RA": f"{data['ra']:.8f}",
            "DEC": f"{data['dec']:.8f}",
            "column": f"{data['x']:.1f}",
            "row": f"{data['y']:.1f}",
            "Variable": var_html,
            "NSS": nss_html,
        }

        if self.extra_cols_in_detail_view is not None:
            for col, label in self.extra_cols_in_detail_view.items():
                key_vals[label] = data.get(col)

        return key_vals, extra_rows


def _join_for_empty_right_table(left, right):
    # astropy.table.join() throws ValueError if 1 or both tables are empty
    # this helper is used to populate the dummy columns
    # so that the resulting table still has all the columns

    rs = left.copy()
    for c in right.colnames:
        if np.issubdtype(right[c].info.dtype.type, str):
            fill_val, dtype = "", str
        elif np.issubdtype(right[c].info.dtype.type, np.integer):
            fill_val, dtype = 0, int
        else:
            fill_val, dtype = np.nan, float
        # use MaskedColumn because that's what astroquery returns
        # (some subsequent codes would use MaskedColumn-specific APIs)
        rs.add_column(
            MaskedColumn(
                np.full_like(rs, fill_val, dtype=dtype),
                name=c,
                mask=np.full_like(rs, True, dtype=bool),
                fill_value=fill_val,
            )
        )
    return rs


class GaiaDR3TICInteractSkyCatalogProvider(GaiaDR3InteractSkyCatalogProvider):
    """
    Provide Gaia DR3 joined with TESS Input Catalog (TIC) data to
    `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    Gaia DR3 data is from Vizier: https://cdsarc.cds.unistra.fr/viz-bin/cat/I/355

    TIC data is from Vizier: https://cdsarc.cds.unistra.fr/viz-bin/cat/IV/39

    The class is used by ``interact_sky()`` internally. The behavior can
    be customized by supplying a dictionary of keyword parameters to
    `catalogs` parameter of ``interact_sky()``. The keyword parameters are
    then used to customize the parameters passed to the constructor here.

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        the coordinate of the target.

    radius: float or `~astropy.units.Quantity`
        the cone search radius, in arc seconds if the value is float.

    magnitude_limit: float
        A value to limit the results in based on Gaia Gmag / TESS Tmag.
        A star is included if Gmag <= magnitude_limit or Tmag <= magnitude_limit.

    scatter_kwargs: dict
        keyword arguments passed to bokeh's ``figure.scatter()``
        function to plot the stars.

    extra_cols_in_detail_view: dict, optional
        additional Gaia DR3 parameters to be included in detail view, in the form
        of (column name in Gaia DR3 Vizier table, display name).

    url_templates: dict, optional
        The URL templates for the URL shown in detail view. The keys are:

        * ``gaiadr3_main_url`` : Gaia DR3 Main
        * ``gaiadr3_var_url`` : Gaia DR3 Variable
        * ``gaiadr3_nss_url`` : Gaia DR3 Non Single Star
        * ``simbad_url_by_gaia_source`` : SIMBAD by Gaia Source
        * ``simbad_url_by_coord`` : SIMBAD by coordinate

        For all except ``simbad_url_by_coord``, the string ``%s`` is replaced by
        Gaia source. For ``simbad_url_by_coord``, the strings ``%ra`` and ``%dec``
        are replaced by the target's right ascension and declination
        (in degrees) respectively.
    """

    # OPEN: composition (with GaiaDR3InteractSkyCatalogProvider as a member, instead of inheriting it)
    # would be cleaner conceptually

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
        extra_cols_in_detail_view: dict = None,
        url_templates: dict = None,
    ) -> None:
        super().__init__(
            coord,
            radius,
            magnitude_limit,
            scatter_kwargs,
            extra_cols_in_detail_view,
            url_templates,
        )
        # TIC-specific
        self.cols_for_source += ["TIC", "Tmag"]
        self.tic_catalog_name = "IV/39/tic82"
        self.exclude_tic_duplicates = True
        self.exclude_tic_artifacts = True

    @property
    def label(self) -> str:
        return "Gaia DR3 + TIC"

    def query_catalog(self) -> Table:
        # do my own custom query so that only one Vizier call is needed to get the result from both Gaia and TIC,
        # reducing time needed.
        with warnings.catch_warnings():
            # suppress useless warning to workaround  https://github.com/astropy/astroquery/issues/2352
            warnings.filterwarnings(  # for Gaia
                "ignore",
                category=u.UnitsWarning,
                message="Unit 'e' not supported by the VOUnit standard",
            )
            warnings.filterwarnings(  # for TIC
                "ignore",
                category=u.UnitsWarning,
                message="Unit 'Sun' not supported by the VOUnit standard",
            )

            gaia_cols = self.columns
            tic_cols = [
                "TIC",
                "GAIA",
                "RAJ2000",
                "DEJ2000",
                "pmRA",
                "pmDE",
                "Plx",
                "Tmag",
                "Disp",
            ]
            rs_list = _query_cone_region(
                self.coord,
                self.radius,
                [self.catalog_name, self.tic_catalog_name],
                columns=gaia_cols + tic_cols,
            )
        gaia_rs = rs_list[self.catalog_name]
        self._post_process_gaiadr3_result(gaia_rs)

        tic_rs = rs_list[self.tic_catalog_name]
        if self.exclude_tic_duplicates:
            # exclude duplicates: 2MASS source split into multiple entries
            # https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes#:~:text=SPLIT%20stars
            tic_rs = tic_rs[tic_rs["Disp"] != "DUPLICATE"]
        if self.exclude_tic_artifacts:
            # exclude artifacts: spurious data (usually from 2MASS)
            # https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes#:~:text=Artifacts%20are%20generally%20spurious
            tic_rs = tic_rs[tic_rs["Disp"] != "ARTIFACT"]

        # Join Gaia and TIC results
        # first do some preparation then join the 2 tables
        # avoid names conflicts in  join
        cols_to_rename = ["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Gmag"]
        tic_rs.rename_columns(
            cols_to_rename,
            [f"t_{c}" for c in cols_to_rename],
        )
        tic_rs["GAIA"] = tic_rs["GAIA"].filled(
            -1
        )  # avoid table merge error (it requires no missing key)
        if len(gaia_rs) > 0 and len(tic_rs) > 0:
            rs = join(
                gaia_rs,
                tic_rs,
                join_type="outer",
                keys_left="Source",
                keys_right="GAIA",
                metadata_conflicts="silent",
            )
        elif len(tic_rs) == 0:
            rs = _join_for_empty_right_table(gaia_rs, tic_rs)
        else:
            rs = _join_for_empty_right_table(tic_rs, gaia_rs)

        #
        # Post-join massaging the data
        #

        # honor mag limit filter with a more liberal policy: either Gmag or Tmag within the limit is okay.
        if self.magnitude_limit is not None:
            rs = rs[
                (rs[self.magnitude_limit_column_name] < self.magnitude_limit)
                | (rs["Tmag"] < self.magnitude_limit)
            ]

        # add column magForSize
        rs["magForSize"] = rs[self.magnitude_limit_column_name]  # Gmag
        rs["magForSize"][rs["Source"].mask] = rs["Tmag"][
            rs["Source"].mask
        ]  # use Tmag when Gaia data is missing

        # handle case missing TIC data
        # make missing integer value as empty string
        for c in ["TIC"]:
            rs[c] = rs[c].astype(str).filled("")
            rs[c].format = None

        # handle cases missing Gaia data
        # use TIC values for columns
        for c in ["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx"]:
            rs[c][rs["Source"].mask] = rs[f"t_{c}"][rs["Source"].mask]
        # make missing integer value as empty string
        for c in ["Source"]:
            rs[c] = rs[c].astype(str).filled("")
            rs[c].format = None

        return rs

    def get_tooltips(self) -> list:
        tooltips = super().get_tooltips()
        return [
            ("TIC", "@TIC"),
            ("Tmag", "@Tmag"),
        ] + tooltips

    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        gaia_key_vals, gaia_extra_rows = super().get_detail_view(data)

        if data["TIC"] != "":
            exofop_url = (
                f'https://exofop.ipac.caltech.edu/tess/target.php?id={data["TIC"]}'
            )
            tic_val_html = (
                f'{data["TIC"]} (<a href="{exofop_url}" target="_blank">ExoFOP</a>)'
            )
        else:
            tic_val_html = "No TIC match (new in DR3 or Gaia ID changed from DR2)"

        key_vals = {
            "TIC": tic_val_html,
            "Tmag": f"{data['Tmag']:.3f}",
        }
        # append Gaia's key-value pairs so they appear after TICs
        key_vals.update(gaia_key_vals)

        return key_vals, gaia_extra_rows
