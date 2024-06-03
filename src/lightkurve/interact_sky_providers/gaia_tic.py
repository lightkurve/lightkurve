from typing import Tuple, Union
import warnings

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astropy.time import Time

from astroquery.vizier import Vizier

from .core import ProperMotionCorrectionMeta, InteractSkyCatalogProvider


def _query_cone_region(coord, radius, catalog, columns=["*"], query_kwargs=dict()) -> Table:
    # Thin wrapper over Vizier's query_region
    vizier = Vizier(
        columns=columns,
    )
    vizier.ROW_LIMIT = -1
    return vizier.query_region(coord, radius=radius, catalog=catalog, **query_kwargs)


class VizierInteractSkyCatalogProvider(InteractSkyCatalogProvider):

    def init(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
    ) -> None:
        super().init(coord, radius, magnitude_limit, scatter_kwargs)
        # Vizier-specific query
        self.catalog_name = None
        self.columns = ["*"]
        self.magnitude_limit_column_name = None

    def query_catalog(self) -> Table:
        with warnings.catch_warnings():
            # suppress useless warning to workaround  https://github.com/astropy/astroquery/issues/2352
            # for Gaia
            warnings.filterwarnings("ignore", category=u.UnitsWarning, message="Unit 'e' not supported by the VOUnit standard")
            result = _query_cone_region(self.coord, self.radius, self.catalog_name, columns=self.columns)
        no_targets_found_message = ValueError("Either no sources were found in the query region " "or Vizier is unavailable")
        too_few_found_message = ValueError("No sources found brighter than {:0.1f}".format(self.magnitude_limit))
        if result is None:
            raise no_targets_found_message
        elif len(result) == 0:
            raise too_few_found_message
        result = result[self.catalog_name]
        if self.magnitude_limit_column_name is not None and self.magnitude_limit is not None:
            result = result[result[self.magnitude_limit_column_name] < self.magnitude_limit]
        if len(result) == 0:
            raise no_targets_found_message

        # to be used as the basis for sizing the dots in plots
        if self.magnitude_limit_column_name is not None:
            result["magForSize"] = result[self.magnitude_limit_column_name]

        return result


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


class GaiaDR3InteractSkyCatalogProvider(VizierInteractSkyCatalogProvider):

    label: str = "Gaia DR3"

    # Gaia DR3 reference epoch: 2016.0,  time coordinate: barycentric coordinate time (TCB).
    # https://www.cosmos.esa.int/web/gaia/dr3
    # J2016 = Time(2016.0, format="jyear", scale="tcb")

    # OPEN: unsure if the epoch RAJ200/DEJ200 is in tt or possibly tdb, etc.
    J2000 = Time(2000.0, format="jyear", scale="tt")

    def init(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
        extra_cols_in_detail_view: dict = None,
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
        super().init(coord, radius, magnitude_limit, scatter_kwargs)
        # Gaia DR3 Vizier specific
        self.catalog_name = "I/355/gaiadr3"
        self.columns = ["*", "RAJ2000", "DEJ2000", "VarFlag", "NSS"]
        self.magnitude_limit_column_name = "Gmag"
        self.cols_for_source = [
            "Source",
            "Gmag",
            "Plx",
            "VarFlag",
            "NSS",
        ]
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

    def query_catalog(self) -> Table:
        tab = super().query_catalog()

        # set custom fill_value for some columns, typically integer columns,
        # so that for rows with missing values, the custom `fill_value` is used
        # for column NSS, without setting fill_value, the astropy default `fill_value`
        # is often 63, confusing  users.
        if "NSS" in tab.colnames:
            tab["NSS"].fill_value = 0

        return tab

    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        # Use RAJ200/ DEJ2000 instead of Gaia DR3's native RA_IRCS in J2016.0 for ease of
        # merging with the result from TIC
        return ProperMotionCorrectionMeta("RAJ2000", "DEJ2000", "pmRA", "pmDE", self.J2000)

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
        vizier_url = (
            "https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-source=+I%2F355%2Fgaiadr3+I%2F355%2Fparamp"
            f"&Source={data['Source']}"
        )
        simbad_url_by_gaia_source = f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident=Gaia DR3 {data['Source']}"
        simbad_url_by_coord = (
            f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={data['ra']}+{data['dec']}&Radius=2&Radius.unit=arcmin"
        )
        if data["Source"] != "":
            source_val_html = f"""{data['Source']} (<a href="{vizier_url}" target="_blank">Vizier</a>)"""
            extra_rows = [
                f'<a target="_blank" href="{simbad_url_by_gaia_source}">SIMBAD by Gaia Source</a>',
                f'<a target="_blank" href="{simbad_url_by_coord}">SIMBAD by coordinate</a>',
            ]
        else:
            source_val_html = ""
            extra_rows = []

        var_html = data["VarFlag"]
        if var_html == "VARIABLE":
            gaiadr3_var_url = (
                "https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-source=+I%2F358%2Fvarisum+I%2F358%2Fvclassre"
                f"&Source={data['Source']}"
            )
            var_html += f' (<a href="{gaiadr3_var_url}" target="_blank">Vizier</a>)'

        nss_html = str(data["NSS"])
        if data["NSS"] != 0:
            flags_text = ", ".join(_decode_gaiadr3_nss_flag(data["NSS"]))
            gaiadr3_nss_url = (
                "https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-ref=VIZ65a1a2351812e4&-source=I%2F357"
                f"&Source={data['Source']}"
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


class GaiaDR3TICInteractSkyCatalogProvider(GaiaDR3InteractSkyCatalogProvider):
    # OPEN: composition (with GaiaDR3InteractSkyCatalogProvider as a member, instead of inheriting it)
    # would be cleaner conceptually

    label: str = "Gaia DR3 + TIC"

    def init(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
        extra_cols_in_detail_view: dict = None,
    ) -> None:
        super().init(coord, radius, magnitude_limit, scatter_kwargs, extra_cols_in_detail_view)
        # TIC-specific
        self.cols_for_source += ["TIC", "Tmag"]
        self.tic_catalog_name = "IV/39/tic82"
        self.exclude_tic_duplicates = True
        self.exclude_tic_artifacts = True

    def query_catalog(self) -> Table:
        gaia_rs = super().query_catalog()

        with warnings.catch_warnings():
            # suppress useless warning to workaround  https://github.com/astropy/astroquery/issues/2352
            # for Gaia
            warnings.filterwarnings(
                "ignore", category=u.UnitsWarning, message="Unit 'Sun' not supported by the VOUnit standard"
            )
            tic_rs = _query_cone_region(
                self.coord,
                self.radius,
                self.tic_catalog_name,
                columns=["TIC", "GAIA", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Tmag", "Disp"],
            )[self.tic_catalog_name]
        if self.exclude_tic_duplicates:
            # exclude duplicates: 2MASS source split into multiple entries
            # https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes#:~:text=SPLIT%20stars
            tic_rs = tic_rs[tic_rs["Disp"] != "DUPLICATE"]
        if self.exclude_tic_artifacts:
            # exclude artifacts: spurious data (usually from 2MASS)
            # https://outerspace.stsci.edu/display/TESS/TIC+v8.2+and+CTL+v8.xx+Data+Release+Notes#:~:text=Artifacts%20are%20generally%20spurious
            tic_rs = tic_rs[tic_rs["Disp"] != "ARTIFACT"]
        if self.magnitude_limit is not None:
            tic_rs = tic_rs[tic_rs["Tmag"] < self.magnitude_limit]

        # Do some preparation then join the 2 tables
        # avoid names conflicts in  join
        cols_to_rename = ["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx"]
        tic_rs.rename_columns(
            cols_to_rename,
            [f"t_{c}" for c in cols_to_rename],
        )
        tic_rs["GAIA"] = tic_rs["GAIA"].filled(-1)  # avoid table merge error (it requires  no missing key)
        rs = join(gaia_rs, tic_rs, join_type="outer", keys_left="Source", keys_right="GAIA", metadata_conflicts="silent")

        # Post-join massaging the data
        # handle case missing TIC
        # make missing integer value as empty string
        for c in ["TIC"]:
            rs[c] = rs[c].astype(str).filled("")
            rs[c].format = None

        # handle cases missing Gaia
        # use TIC values for columns
        for c in ["RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx"]:
            rs[c][rs["Source"].mask] = rs[f"t_{c}"][rs["Source"].mask]
        rs["magForSize"][rs["Source"].mask] = rs["Tmag"][rs["Source"].mask]  # use Tmag when Gaia data is missing
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
            exofop_url = f'https://exofop.ipac.caltech.edu/tess/target.php?id={data["TIC"]}'
            tic_val_html = f'{data["TIC"]} (<a href="{exofop_url}" target="_blank">ExoFOP</a>)'
        else:
            tic_val_html = "No TIC match (new in DR3 or Gaia ID changed from DR2)"

        key_vals = {
            "TIC": tic_val_html,
            "Tmag": f"{data['Tmag']:.3f}",
        }
        # append Gaia's key-value pairs so they appear after TICs
        key_vals.update(gaia_key_vals)

        return key_vals, gaia_extra_rows
