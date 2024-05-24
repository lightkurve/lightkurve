"""Provides tools for interactive visualizations.

Example use
-----------
The functions in this module are used to create Bokeh-based visualization
widgets.  For example, the following code will create an interactive
visualization widget showing the pixel data and a lightcurve::

    # SN 2018 oh Supernova example
    from lightkurve import KeplerTargetPixelFile
    tpf = KeplerTargetPixelFile.from_archive(228682548)
    tpf.interact()

Note that this will only work inside a Jupyter notebook at this time.
"""
from __future__ import division, print_function
import os
import logging
import warnings

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astropy.io import ascii
from astropy.stats import sigma_clip
from astropy.time import Time
import astropy.units as u
from astropy.utils.exceptions import AstropyUserWarning
import pandas as pd
from pandas import Series


from .utils import KeplerQualityFlags, LightkurveWarning, LightkurveError, finalize_notebook_url

log = logging.getLogger(__name__)

# Import the optional Bokeh dependency, or print a friendly error otherwise.
_BOKEH_IMPORT_ERROR = None
try:
    import bokeh  # Import bokeh first so we get an ImportError we can catch
    from bokeh.io import show, output_notebook
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.models import (
        LogColorMapper,
        Slider,
        RangeSlider,
        Span,
        ColorBar,
        LogTicker,
        Range1d,
        LinearColorMapper,
        BasicTicker,
        Arrow,
        VeeHead,
    )
    from bokeh.layouts import layout, Spacer
    from bokeh.models.tools import HoverTool
    from bokeh.models.widgets import Button, Div
    from bokeh.models.formatters import PrintfTickFormatter
except Exception as e:
    # We will print a nice error message in the `show_interact_widget` function
    # the error would be raised there in case users need to diagnose problems
    _BOKEH_IMPORT_ERROR = e


def _search_nearby_of_tess_target(tic_id):
    # To avoid warnings / overflow error in attempting to convert GAIA DR2, TIC ID, TOI
    # as int32 (the default) in some cases
    return ascii.read(f"https://exofop.ipac.caltech.edu/tess/download_nearbytarget.php?id={tic_id}&output=csv",
                      format="csv",
                      fast_reader=False,
                      converters={
                          "GAIA DR2": [ascii.convert_numpy(str)],
                          "TIC ID": [ascii.convert_numpy(str)],
                          "TOI": [ascii.convert_numpy(str)],
                          })


def _get_tic_meta_of_gaia_in_nearby(tab, nearby_gaia_id, key, default=None):
    res = tab[tab['GAIA DR2'] == str(nearby_gaia_id)]
    if len(res) > 0:
        return res[0][key]
    else:
        return default


def _correct_with_proper_motion(ra, dec, pm_ra, pm_dec, equinox, new_time):
    """Return proper-motion corrected RA / Dec.
       It also return whether proper motion correction is applied or not."""
    # all parameters have units

    if ra is None or dec is None or \
       pm_ra is None or pm_dec is None or (np.all(pm_ra == 0) and np.all(pm_dec == 0)) or \
       equinox is None:
        return ra, dec, False

    # To be more accurate, we should have supplied distance to SkyCoord
    # in theory, for Gaia DR2 data, we can infer the distance from the parallax provided.
    # It is not done for 2 reasons:
    # 1. Gaia DR2 data has negative parallax values occasionally. Correctly handling them could be tricky. See:
    #    https://www.cosmos.esa.int/documents/29201/1773953/Gaia+DR2+primer+version+1.3.pdf/a4459741-6732-7a98-1406-a1bea243df79
    # 2. For our purpose (ploting in various interact usage) here, the added distance does not making
    #    noticeable significant difference. E.g., applying it to Proxima Cen, a target with large parallax
    #    and huge proper motion, does not change the result in any noticeable way.
    #
    c = SkyCoord(ra, dec, pm_ra_cosdec=pm_ra, pm_dec=pm_dec,
                frame='icrs', obstime=equinox)

    # Suppress ErfaWarning temporarily as a workaround for:
    #   https://github.com/astropy/astropy/issues/11747
    with warnings.catch_warnings():
        # the same warning appears both as an ErfaWarning and a astropy warning
        # so we filter by the message instead
        warnings.filterwarnings("ignore", message="ERFA function")
        new_c = c.apply_space_motion(new_obstime=new_time)
    return new_c.ra, new_c.dec, True


def _get_corrected_coordinate(tpf_or_lc):
    """Extract coordinate from Kepler/TESS FITS, with proper motion corrected
       to the start of observation if proper motion is available."""
    h = tpf_or_lc.meta
    new_time = tpf_or_lc.time[0]

    ra = h.get("RA_OBJ")
    dec = h.get("DEC_OBJ")

    pm_ra = h.get("PMRA")
    pm_dec = h.get("PMDEC")
    equinox = h.get("EQUINOX")

    if ra is None or dec is None or pm_ra is None or pm_dec is None or equinox is None:
        # case cannot apply proper motion due to missing parameters
        return ra, dec, False

    # Note: it'd be better / extensible if the unit is a property of the tpf or lc
    if tpf_or_lc.meta.get("TICID") is not None:
        pm_unit = u.milliarcsecond / u.year
    else:  # assumes to be Kepler / K2
        pm_unit = u.arcsecond / u.year

    ra_corrected, dec_corrected, pm_corrected = _correct_with_proper_motion(
            ra * u.deg, dec *u.deg,
            pm_ra * pm_unit, pm_dec * pm_unit,
            # we assume the data is in J2000 epoch
            Time('2000', format='byear'),
            new_time)
    return ra_corrected.to(u.deg).value,  dec_corrected.to(u.deg).value, pm_corrected


def _to_unitless(items):
    """Convert the values in the item list to unitless one"""
    return [getattr(item, "value", item) for item in items]


def prepare_lightcurve_datasource(lc):
    """Prepare a bokeh ColumnDataSource object for tool tips.

    Parameters
    ----------
    lc : LightCurve object
        The light curve to be shown.

    Returns
    -------
    lc_source : bokeh.plotting.ColumnDataSource
    """
    # Convert time into human readable strings, breaks with NaN time
    # See https://github.com/lightkurve/lightkurve/issues/116
    if (lc.time == lc.time).all():
        human_time = lc.time.isot
    else:
        human_time = [" "] * len(lc.flux)

    # Convert binary quality numbers into human readable strings
    qual_strings = []
    for bitmask in lc.quality:
        if isinstance(bitmask, u.Quantity):
            bitmask = bitmask.value
        flag_str_list = KeplerQualityFlags.decode(bitmask)
        if len(flag_str_list) == 0:
            qual_strings.append(" ")
        if len(flag_str_list) == 1:
            qual_strings.append(flag_str_list[0])
        if len(flag_str_list) > 1:
            qual_strings.append("; ".join(flag_str_list))

    lc_source = ColumnDataSource(
        data=dict(
            time=lc.time.value,
            time_iso=human_time,
            flux=lc.flux.value,
            cadence=lc.cadenceno.value,
            quality_code=lc.quality.value,
            quality=np.array(qual_strings),
        )
    )
    return lc_source


def aperture_mask_to_selected_indices(aperture_mask):
    """Convert the 2D aperture mask to 1D selection indices, for the use with bokeh ColumnDataSource."""
    npix = aperture_mask.size
    pixel_index_array = np.arange(0, npix, 1)
    return pixel_index_array[aperture_mask.reshape(-1)]


def aperture_mask_from_selected_indices(selected_pixel_indices, tpf):
    """Convert an aperture mask in 1D selection indices back to 2D (in the shape of the given TPF)."""
    npix = tpf.flux[0, :, :].size
    pixel_index_array = np.arange(0, npix, 1).reshape(tpf.flux[0].shape)
    selected_indices = np.array(selected_pixel_indices)
    selected_mask_1d = np.isin(pixel_index_array, selected_indices)
    return selected_mask_1d.reshape(tpf.flux[0].shape)


def prepare_tpf_datasource(tpf, aperture_mask):
    """Prepare a bokeh DataSource object for selection glyphs

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to be shown.
    aperture_mask : boolean numpy array
        The Aperture mask applied at the startup of interact

    Returns
    -------
    tpf_source : bokeh.plotting.ColumnDataSource
        Bokeh object to be shown.
    """
    _, ny, nx = tpf.shape
    # (xa, ya) pair enumerates all pixels of the tpf
    xx = tpf.column + np.arange(nx)
    yy = tpf.row + np.arange(ny)
    xa, ya = np.meshgrid(xx, yy)
    # flatten them, as column data source requires 1d data
    xa = xa.flatten()
    ya = ya.flatten()
    tpf_source = ColumnDataSource(data=dict(xx=xa.astype(float), yy=ya.astype(float)))
    # convert the ndarray from aperture_mask_to_selected_indices() to plain list
    # because bokeh v3.0.2 does not accept ndarray (and causes js error)
    # see https://github.com/bokeh/bokeh/issues/12624
    tpf_source.selected.indices = list(aperture_mask_to_selected_indices(aperture_mask))
    return tpf_source


def get_lightcurve_y_limits(lc_source):
    """Compute sensible defaults for the Y axis limits of the lightcurve plot.

    Parameters
    ----------
    lc_source : bokeh.plotting.ColumnDataSource
        The lightcurve being shown.

    Returns
    -------
    ymin, ymax : float, float
        Flux min and max limits.
    """
    with warnings.catch_warnings():  # Ignore warnings due to NaNs
        warnings.simplefilter("ignore", AstropyUserWarning)
        flux = sigma_clip(lc_source.data["flux"], sigma=5, masked=False)
    low, high = np.nanpercentile(flux, (1, 99))
    margin = 0.10 * (high - low)
    return low - margin, high + margin


def make_lightcurve_figure_elements(lc, lc_source, ylim_func=None):
    """Make the lightcurve figure elements.

    Parameters
    ----------
    lc : LightCurve
        Lightcurve to be shown.
    lc_source : bokeh.plotting.ColumnDataSource
        Bokeh object that enables the visualization.

    Returns
    ----------
    fig : `bokeh.plotting.figure` instance
    step_renderer : GlyphRenderer
    vertical_line : Span
    """
    mission = lc.meta.get("MISSION")
    if mission == "K2":
        title = "Lightcurve for {} (K2 C{})".format(lc.label, lc.campaign)
    elif mission == "Kepler":
        title = "Lightcurve for {} (Kepler Q{})".format(lc.label, lc.quarter)
    elif mission == "TESS":
        title = "Lightcurve for {} (TESS Sec. {})".format(lc.label, lc.sector)
    else:
        title = "Lightcurve for target {}".format(lc.label)

    fig = figure(
        title=title,
        height=340,
        width=600,
        tools="pan,wheel_zoom,box_zoom,tap,reset",
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )
    fig.title.offset = -10

    # ylabel: mimic the logic in lc._create_plot()
    ylabel = "Flux"
    if lc.meta.get("NORMALIZED"):
        ylabel = "Normalized " + ylabel
    elif (lc["flux"].unit) and (lc["flux"].unit.to_string() != ""):
        if lc["flux"].unit == (u.electron / u.second):
            yunit_str = "e/s"  # the common case, use abbreviation
        else:
            yunit_str = lc["flux"].unit.to_string()
        ylabel += f" ({yunit_str})"
    fig.yaxis.axis_label = ylabel

    fig.xaxis.axis_label = "Time (days)"
    try:
        if (lc.mission == "K2") or (lc.mission == "Kepler"):
            fig.xaxis.axis_label = "Time - 2454833 (days)"
        elif lc.mission == "TESS":
            fig.xaxis.axis_label = "Time - 2457000 (days)"
    except AttributeError:  # no mission keyword available
        pass

    if ylim_func is None:
        ylims = get_lightcurve_y_limits(lc_source)
    else:
        ylims = _to_unitless(ylim_func(lc))
    fig.y_range = Range1d(start=ylims[0], end=ylims[1])

    # Add step lines, circles, and hover-over tooltips
    fig.step(
        "time",
        "flux",
        line_width=1,
        color="gray",
        source=lc_source,
        nonselection_line_color="gray",
        nonselection_line_alpha=1.0,
    )
    circ = fig.scatter(
        "time",
        "flux",
        source=lc_source,
        fill_alpha=0.3,
        size=8,
        line_color=None,
        selection_color="firebrick",
        nonselection_fill_alpha=0.0,
        nonselection_fill_color="grey",
        nonselection_line_color=None,
        nonselection_line_alpha=0.0,
        fill_color=None,
        hover_fill_color="firebrick",
        hover_alpha=0.9,
        hover_line_color="white",
    )
    tooltips = [
        ("Cadence", "@cadence"),
        ("Time ({})".format(lc.time.format.upper()), "@time{0,0.000}"),
        ("Time (ISO)", "@time_iso"),
        ("Flux", "@flux"),
        ("Quality Code", "@quality_code"),
        ("Quality Flag", "@quality"),
    ]
    fig.add_tools(
        HoverTool(
            tooltips=tooltips,
            renderers=[circ],
            mode="mouse",
            point_policy="snap_to_data",
        )
    )

    # Vertical line to indicate the cadence
    vertical_line = Span(
        location=lc.time[0].value,
        dimension="height",
        line_color="firebrick",
        line_width=4,
        line_alpha=0.5,
    )
    fig.add_layout(vertical_line)

    return fig, vertical_line


def _add_tics_with_matching_gaia_ids_to(result, tab, gaia_ids):
    # use pandas Series rather than plain list, so they look like the existing columns in the source
    #
    # Note: we convert all the data to string to better handles cases when a star has no TIC
    # In such cases, if we supply None as a value in a pandas Series,
    # bokeh's tooltip template will render it as NaN (rather than empty string)
    # To avoid NaN display, we force the Series to use string dtype, and for stars with missing TICs,
    # empty string will be used as the value. bokeh's tooltip template can correctly render it as empty string

    col_tic_id = Series(data=[_get_tic_meta_of_gaia_in_nearby(tab, id, 'TIC ID', "") for id in gaia_ids],
                        dtype=str)
    col_tess_mag = Series(data=[_get_tic_meta_of_gaia_in_nearby(tab, id, 'TESS Mag', "") for id in gaia_ids],
                          dtype=str)
    col_separation = Series(data=[_get_tic_meta_of_gaia_in_nearby(tab, id, 'Separation (arcsec)', "") for id in gaia_ids],
                            dtype=str)

    result['tic'] = col_tic_id
    result['TESSmag'] = col_tess_mag
    result['separation'] = col_separation
    return result

# use case: signify Gaia ID (Source, int type) as missing
_MISSING_INT_VAL = 0

def _add_tics_with_no_matching_gaia_ids_to(result, tab, gaia_ids, magnitude_limit):
    def _add_to(data_dict, dest_colname, src):
        # the data_dict should ultimately have the same columns/dtype as the result,
        # as it will be appended to the result at the end
        data_dict[dest_colname] = Series(data=src, dtype=result[dest_colname].dtype)

    def _dummy_like(ary, dtype):
        dummy_val = None
        if pd.api.types.is_integer_dtype(dtype):
            dummy_val = _MISSING_INT_VAL
        elif pd.api.types.is_float_dtype(dtype):
            dummy_val = np.nan
        return [dummy_val for i in range(len(ary))]

    # filter out those with matching gaia ids
    # (handled in `_add_tics_with_matching_gaia_ids_to()`)
    gaia_str_ids = [str(id) for id in gaia_ids]
    tab = tab[np.isin(tab['GAIA DR2'], gaia_str_ids, invert=True)]

    # filter out those with gaia ids, but Gaia Mag is smaller than magnitude_limit
    # (they won't appear in the given gaia_ids list)
    tab = tab[tab['GAIA Mag'] < magnitude_limit]

    # apply magnitude_limit filter for those with no Gaia data using TESS mag
    tab = tab[tab['TESS Mag'] < magnitude_limit]

    # convert the filtered tab to a dataframe, so as to append to the existing result
    data = dict()
    _add_to(data, 'tic', tab['TIC ID'])
    _add_to(data, 'TESSmag', tab['TESS Mag'])
    _add_to(data, 'magForSize', tab['TESS Mag'])
    _add_to(data, 'separation', tab['Separation (arcsec)'])
    # convert the string Ra/Dec to float
    # we assume the equinox is the same as those from Gaia DR2
    coords = SkyCoord(tab['RA'], tab['Dec'], unit=(u.hourangle, u.deg), frame='icrs')
    _add_to(data, 'RA_ICRS', coords.ra.value)
    _add_to(data, 'DE_ICRS', coords.dec.value)
    _add_to(data, 'pmRA', tab['PM RA (mas/yr)'])
    _add_to(data, 'e_pmRA', tab['PM RA Err (mas/yr)'])
    _add_to(data, 'pmDE', tab['PM Dec (mas/yr)'])
    _add_to(data, 'e_pmDE', tab['PM Dec Err (mas/yr)'])

    # add dummy columns so that the resulting data frame would match the existing one
    nontic_colnames = [c for c in result.keys() if c not in data.keys()]
    for c in nontic_colnames:
        data[c] = Series(data=_dummy_like(tab, result[c].dtype), dtype=result[c].dtype)

    # finally, append the entries to existing result dataframe
    return pd.concat([result, pd.DataFrame(data)])


def _add_nearby_tics_if_tess(tpf, magnitude_limit, result):
    tic_id = tpf.meta.get('TICID', None)
    # handle 3 cases:
    # - TESS tpf has a valid id, type integer
    # - Some TESSCut has empty string while and some others has None
    # - Kepler tpf does not have the header
    if tic_id is None or tic_id == "":
        return result, []

    if isinstance(tic_id, str):
        # for cases tpf is from tpf.cutout() call in #1089
        tic_id = tic_id.replace("_CUTOUT", "")

    # nearby TICs from ExoFOP
    tab = _search_nearby_of_tess_target(tic_id)
    gaia_ids = result['Source'].array

    # merge the TICs with matching Gaia entries
    result = _add_tics_with_matching_gaia_ids_to(result, tab, gaia_ids)

    # add new entries for the TICs with no matching Gaia ones
    result = _add_tics_with_no_matching_gaia_ids_to(result, tab, gaia_ids, magnitude_limit)

    source_colnames_extras = ['tic', 'TESSmag', 'separation']
    tooltips_extras = [("TIC", "@tic"), ("TESS Mag", "@TESSmag"), ("Separation (\")", "@separation")]
    return result, source_colnames_extras, tooltips_extras


def _to_display(series):
    def _format(val):
        if val == _MISSING_INT_VAL or np.isnan(val):
            return ""
        else:
            return str(val)
    return pd.Series(data=[_format(v) for v in series], dtype=str)


def _get_nearby_gaia_objects(tpf, magnitude_limit=18):
    """Get nearby objects (of the target defined in tpf) from Gaia.
    The result is formatted for the use of plot."""
    # Get the positions of the Gaia sources
    try:
        c1 = SkyCoord(tpf.ra, tpf.dec, frame="icrs", unit="deg")
    except Exception as err:
        msg = ("Cannot get nearby stars in GAIA because TargetPixelFile has no valid coordinate. "
               f"ra: {tpf.ra}, dec: {tpf.dec}")
        raise LightkurveError(msg) from err

    # Use pixel scale for query size
    pix_scale = 4.0  # arcseconds / pixel for Kepler, default
    if tpf.mission == "TESS":
        pix_scale = 21.0
    # We are querying with a diameter as the radius, overfilling by 2x.
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = -1
    with warnings.catch_warnings():
        # suppress useless warning to workaround  https://github.com/astropy/astroquery/issues/2352
        warnings.filterwarnings(
            "ignore", category=u.UnitsWarning, message="Unit 'e' not supported by the VOUnit standard"
        )
        result = Vizier.query_region(
            c1,
            catalog=["I/345/gaia2"],
            radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"),
        )
    no_targets_found_message = ValueError(
        "Either no sources were found in the query region " "or Vizier is unavailable"
    )
    too_few_found_message = ValueError(
        "No sources found brighter than {:0.1f}".format(magnitude_limit)
    )
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise too_few_found_message
    result = result["I/345/gaia2"].to_pandas()
    result = result[result.Gmag < magnitude_limit]
    if len(result) == 0:
        raise no_targets_found_message
    # drop all the filtered rows, it makes subsequent TESS-specific processing easier (to add rows/columns)
    result.reset_index(drop=True, inplace=True)
    result['magForSize'] = result['Gmag']  # to be used as the basis for sizing the dots in plots
    return result


def add_gaia_figure_elements(tpf, fig, magnitude_limit=18):
    """Make the Gaia Figure Elements"""

    result = _get_nearby_gaia_objects(tpf, magnitude_limit)

    source_colnames_extras = []
    tooltips_extras = []
    try:
        result, source_colnames_extras, tooltips_extras = _add_nearby_tics_if_tess(tpf, magnitude_limit, result)
    except Exception as err:
        warnings.warn(
            f"interact_sky() - cannot obtain nearby TICs. Skip it. The error: {err}",
            LightkurveWarning,
        )

    ra_corrected, dec_corrected, _ = _correct_with_proper_motion(
            np.nan_to_num(np.asarray(result.RA_ICRS)) * u.deg, np.nan_to_num(np.asarray(result.DE_ICRS)) * u.deg,
            np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond / u.year,
            np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond / u.year,
            Time(2457206.375, format="jd", scale="tdb"),
            tpf.time[0])
    result.RA_ICRS = ra_corrected.to(u.deg).value
    result.DE_ICRS = dec_corrected.to(u.deg).value
    # Convert to pixel coordinates
    radecs = np.vstack([result["RA_ICRS"], result["DE_ICRS"]]).T
    coords = tpf.wcs.all_world2pix(radecs, 0)

    # Gently size the points by their Gaia magnitude
    sizes = 64.0 / 2 ** (result["magForSize"] / 5.0)
    one_over_parallax = 1.0 / (result["Plx"] / 1000.0)
    source = ColumnDataSource(
        data=dict(
            ra=result["RA_ICRS"],
            dec=result["DE_ICRS"],
            pmra=result["pmRA"],
            pmde=result["pmDE"],
            source=_to_display(result["Source"]),
            Gmag=result["Gmag"],
            plx=result["Plx"],
            one_over_plx=one_over_parallax,
            x=coords[:, 0] + tpf.column,
            y=coords[:, 1] + tpf.row,
            size=sizes,
        )
    )
    for c in source_colnames_extras:
        source.data[c] = result[c]

    tooltips = [
        ("Gaia source", "@source"),
        ("G", "@Gmag"),
        ("Parallax (mas)", "@plx (~@one_over_plx{0,0} pc)"),
        ("RA", "@ra{0,0.00000000}"),
        ("DEC", "@dec{0,0.00000000}"),
        ("pmRA", "@pmra{0,0.000} mas/yr"),
        ("pmDE", "@pmde{0,0.000} mas/yr"),
        ("column", "@x{0.0}"),
        ("row", "@y{0.0}"),
        ]
    tooltips = tooltips_extras + tooltips

    r = fig.scatter(
        "x",
        "y",
        source=source,
        fill_alpha=0.3,
        size="size",
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

    fig.add_tools(
        HoverTool(
            tooltips=tooltips,
            renderers=[r],
            mode="mouse",
            point_policy="snap_to_data",
        )
    )

    # mark the target's position too
    target_ra, target_dec, pm_corrected = _get_corrected_coordinate(tpf)
    target_x, target_y = None, None
    if target_ra is not None and target_dec is not None:
        pix_x, pix_y = tpf.wcs.all_world2pix([(target_ra, target_dec)], 0)[0]
        target_x, target_y = tpf.column + pix_x, tpf.row + pix_y
        fig.scatter(marker="cross", x=target_x, y=target_y, size=20, color="black", line_width=1)
        if not pm_corrected:
            warnings.warn(("Proper motion correction cannot be applied to the target, as none is available. "
                           "Thus the target (the cross) might be noticeably away from its actual position, "
                           "if it has large proper motion."),
                           category=LightkurveWarning)

    # display an arrow on the selected target
    arrow_head = VeeHead(size=16)
    arrow_4_selected = Arrow(end=arrow_head, line_color="red", line_width=4,
                             x_start=0, y_start=0, x_end=0, y_end=0, tags=["selected"],
                             visible=False)
    fig.add_layout(arrow_4_selected)

    def show_arrow_at_target(attr, old, new):
        if len(new) > 0:
            x, y = source.data["x"][new[0]], source.data["y"][new[0]]

            # workaround: the arrow_head color should have been specified once
            # in its creation, but it seems to hit a bokeh bug, resulting in an error
            # of the form  ValueError("expected ..., got {'value': 'red'}")
            # in actual websocket call, it seems that the color value is
            # sent as "{'value': 'red'}", but they are expdecting "red" instead.
            # somehow the error is bypassed if I specify it later in here.
            #
            # The issue is present in bokeh 2.2.3 / 2.1.1, but  not in bokeh 2.3.1
            # I cannot identify a specific issue /PR on github about it though.
            arrow_head.fill_color = "red"
            arrow_head.line_color = "black"

            # place the arrow near (x,y), taking care of boundary cases (at the edge of the plot)
            if x < fig.x_range.start + 1:
                # boundary case: the point is at the left edge of the plot
                arrow_4_selected.x_start = x + 0.85
                arrow_4_selected.x_end = x + 0.2
            elif x > fig.x_range.end - 1:
                # boundary case: the point is at the right edge of the plot
                arrow_4_selected.x_start = x - 0.85
                arrow_4_selected.x_end = x - 0.2
            elif target_x is None or x < target_x:
                # normal case 1 : point is to the left of the target
                arrow_4_selected.x_start = x - 0.85
                arrow_4_selected.x_end = x - 0.2
            else:
                # normal case 2 : point is to the right of the target
                # flip arrow's direction so that it won't overlap with the target
                arrow_4_selected.x_start = x + 0.85
                arrow_4_selected.x_end = x + 0.2

            if y > fig.y_range.end - 0.5:
                # boundary case: the point is at near the top of the plot
                arrow_4_selected.y_start = y - 0.4
                arrow_4_selected.y_end = y - 0.1
            elif y < fig.y_range.start + 0.5:
                # boundary case: the point is at near the top of the plot
                arrow_4_selected.y_start = y + 0.4
                arrow_4_selected.y_end = y + 0.1
            else:  # normal case
                arrow_4_selected.y_start = y
                arrow_4_selected.y_end = y

            arrow_4_selected.visible = True
        else:
            arrow_4_selected.visible = False

    source.selected.on_change("indices", show_arrow_at_target)


    # a widget that displays some of the selected star's metadata
    # so that they can be copied (e.g., GAIA ID).
    # It is a workaround, because bokeh's hover tooltip disappears as soon as the mouse is away from the star.
    message_selected_target = Div(text="")

    def show_target_info(attr, old, new):
        # the following is essentially redoing the bokeh tooltip template above in plain HTML
        # with some slight tweak, mainly to add some helpful links.
        #
        # Note: in source, columns "x" and "y" are ndarray while other column are pandas Series,
        # so the access api is slightly different.
        if len(new) > 0:
            msg = "Selected:<br><table>"
            for idx in new:
                tic_id = source.data['tic'].iat[idx] if source.data.get('tic') is not None else None
                if tic_id is not None and tic_id != "":  # TESS-specific meta data, if available
                    msg += f"""
<tr><td>TIC</td><td>{tic_id}
(<a target="_blank" href="https://exofop.ipac.caltech.edu/tess/target.php?id={tic_id}">ExoFOP</a>)</td></tr>
<tr><td>TESS Mag</td><td>{source.data['TESSmag'].iat[idx]}</td></tr>
<tr><td>Separation (")</td><td>{source.data['separation'].iat[idx]}</td></tr>
"""
                # the main meta data
                msg += f"""
<tr><td>Gaia source</td><td>{source.data['source'].iat[idx]}
(<a target="_blank"
    href="http://vizier.u-strasbg.fr/viz-bin/VizieR-S?Gaia DR2 {source.data['source'].iat[idx]}">Vizier</a>)</td></tr>
<tr><td>G</td><td>{source.data['Gmag'].iat[idx]:.3f}</td></tr>
<tr><td>Parallax (mas)</td>
    <td>{source.data['plx'].iat[idx]:,.3f} (~ {source.data['one_over_plx'].iat[idx]:,.0f} pc)</td>
</tr>
<tr><td>RA</td><td>{source.data['ra'].iat[idx]:,.8f}</td></tr>
<tr><td>DEC</td><td>{source.data['dec'].iat[idx]:,.8f}</td></tr>
<tr><td>pmRA</td><td>{source.data['pmra'].iat[idx]} mas/yr</td></tr>
<tr><td>pmDE</td><td>{source.data['pmde'].iat[idx]} mas/yr</td></tr>
<tr><td>column</td><td>{source.data['x'][idx]:.1f}</td></tr>
<tr><td>row</td><td>{source.data['y'][idx]:.1f}</td></tr>
<tr><td colspan="2">Search
<a target="_blank"
   href="http://simbad.u-strasbg.fr/simbad/sim-id?Ident=Gaia DR2 {source.data['source'].iat[idx]}">
SIMBAD by Gaia ID</a></td></tr>
<tr><td colspan="2">
<a target="_blank"
   href="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={source.data['ra'].iat[idx]}+{source.data['dec'].iat[idx]}&Radius=2&Radius.unit=arcmin">
SIMBAD by coordinate</a></td></tr>
<tr><td colspan="2">&nbsp;</td></tr>
"""

            msg += "\n<table>"
            message_selected_target.text = msg
        # else do nothing (not clearing the widget) for now.

    def on_selected_change(*args):
        show_arrow_at_target(*args)
        show_target_info(*args)

    source.selected.on_change("indices", show_target_info)

    return fig, r, message_selected_target


def to_selected_pixels_source(tpf_source):
    xx = tpf_source.data["xx"].flatten()
    yy = tpf_source.data["yy"].flatten()
    selected_indices = tpf_source.selected.indices
    return ColumnDataSource(dict(
        xx=xx[selected_indices],
        yy=yy[selected_indices],
    ))


def make_tpf_figure_elements(
    tpf,
    tpf_source,
    tpf_source_selectable=True,
    pedestal=None,
    fiducial_frame=None,
    width=370,
    height=340,
    scale="log",
    vmin=None,
    vmax=None,
    cmap="Viridis256",
    tools="tap,box_select,wheel_zoom,reset",
):
    """Returns the lightcurve figure elements.

    Parameters
    ----------
    tpf : TargetPixelFile
        TPF to show.
    tpf_source : bokeh.plotting.ColumnDataSource
        TPF data source.
    tpf_source_selectable : boolean
        True if the tpf_source is selectable. False to show the selected pixels
        in the tpf_source only. Default is True.
    pedestal: float
        A scalar value to be added to the TPF flux values, often to avoid
        taking the log of a negative number in colorbars.
        Defaults to `-min(tpf.flux) + 1`
    fiducial_frame: int
        The tpf slice to start with by default, it is assumed the WCS
        is exact for this frame.
    scale: str
        Color scale for tpf figure. Default is 'log'
    vmin: int [optional]
        Minimum color scale for tpf figure
    vmax: int [optional]
        Maximum color scale for tpf figure
    cmap: str
        Colormap to use for tpf plot. Default is 'Viridis256'
    tools: str
        Bokeh tool list
    Returns
    -------
    fig, stretch_slider : bokeh.plotting.figure.Figure, RangeSlider
    """
    if pedestal is None:
        pedestal = -np.nanmin(tpf.flux.value) + 1
    if scale == "linear":
        pedestal = 0

    if tpf.mission in ["Kepler", "K2"]:
        title = "Pixel data (CCD {}.{})".format(tpf.module, tpf.output)
    elif tpf.mission == "TESS":
        title = "Pixel data (Camera {}.{})".format(tpf.camera, tpf.ccd)
    else:
        title = "Pixel data"

    # We subtract 0.5 from the range below because pixel coordinates refer to
    # the middle of a pixel, e.g. (col, row) = (10.0, 20.0) is a pixel center.
    fig = figure(
        width=width,
        height=height,
        x_range=(tpf.column - 0.5, tpf.column + tpf.shape[2] - 0.5),
        y_range=(tpf.row - 0.5, tpf.row + tpf.shape[1] - 0.5),
        title=title,
        tools=tools,
        toolbar_location="below",
        border_fill_color="whitesmoke",
    )

    fig.yaxis.axis_label = "Pixel Row Number"
    fig.xaxis.axis_label = "Pixel Column Number"

    vlo, lo, hi, vhi = np.nanpercentile(tpf.flux.value + pedestal, [0.2, 1, 95, 99.8])
    if vmin is not None:
        vlo, lo = vmin, vmin
    if vmax is not None:
        vhi, hi = vmax, vmax

    if scale == "log":
        vstep = (np.log10(vhi) - np.log10(vlo)) / 300.0  # assumes counts >> 1.0!
    if scale == "linear":
        vstep = (vhi - vlo) / 300.0  # assumes counts >> 1.0!

    if scale == "log":
        color_mapper = LogColorMapper(palette=cmap, low=lo, high=hi)
    elif scale == "linear":
        color_mapper = LinearColorMapper(palette=cmap, low=lo, high=hi)
    else:
        raise ValueError("Please specify either `linear` or `log` scale for color.")

    fig.image(
        [tpf.flux.value[fiducial_frame, :, :] + pedestal],
        x=tpf.column - 0.5,
        y=tpf.row - 0.5,
        dw=tpf.shape[2],
        dh=tpf.shape[1],
        dilate=True,
        color_mapper=color_mapper,
        name="tpfimg",
    )

    # The colorbar will update with the screen stretch slider
    # The colorbar margin increases as the length of the tick labels grows.
    # This colorbar share of the plot window grows, shrinking plot area.
    # This effect is known, some workarounds might work to fix the plot area:
    # https://github.com/bokeh/bokeh/issues/5186

    if scale == "log":
        ticker = LogTicker(desired_num_ticks=8)
    elif scale == "linear":
        ticker = BasicTicker(desired_num_ticks=8)

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=ticker,
        label_standoff=-10,
        border_line_color=None,
        location=(0, 0),
        background_fill_color="whitesmoke",
        major_label_text_align="left",
        major_label_text_baseline="middle",
        title="e/s",
        margin=0,
    )
    fig.add_layout(color_bar, "right")

    color_bar.formatter = PrintfTickFormatter(format="%14i")

    if tpf_source is not None:
        if tpf_source_selectable:
            fig.rect(
                "xx",
                "yy",
                1,
                1,
                source=tpf_source,
                fill_color="gray",
                fill_alpha=0.4,
                line_color="white",
                )
        else:
            # Paint the selected pixels such that they cannot be selected / deselected.
            # Used to show specified aperture pixels without letting users to
            # change them in ``interact_sky```
            selected_pixels_source = to_selected_pixels_source(tpf_source)
            r_selected = fig.rect(
                "xx",
                "yy",
                1,
                1,
                source=selected_pixels_source,
                fill_color="gray",
                fill_alpha=0.0,
                line_color="white",
                )
            r_selected.nonselection_glyph = None

    # Configure the stretch slider and its callback function
    if scale == "log":
        start, end = np.log10(vlo), np.log10(vhi)
        values = (np.log10(lo), np.log10(hi))
    elif scale == "linear":
        start, end = vlo, vhi
        values = (lo, hi)

    stretch_slider = RangeSlider(
        start=start,
        end=end,
        step=vstep,
        title="Screen Stretch ({})".format(scale),
        value=values,
        orientation="horizontal",
        width=200,
        direction="ltr",
        show_value=True,
        sizing_mode="fixed",
        height=15,
        name="tpfstretch",
    )

    def stretch_change_callback_log(attr, old, new):
        """TPF stretch slider callback."""
        fig.select("tpfimg")[0].glyph.color_mapper.high = 10 ** new[1]
        fig.select("tpfimg")[0].glyph.color_mapper.low = 10 ** new[0]

    def stretch_change_callback_linear(attr, old, new):
        """TPF stretch slider callback."""
        fig.select("tpfimg")[0].glyph.color_mapper.high = new[1]
        fig.select("tpfimg")[0].glyph.color_mapper.low = new[0]

    if scale == "log":
        stretch_slider.on_change("value", stretch_change_callback_log)
    if scale == "linear":
        stretch_slider.on_change("value", stretch_change_callback_linear)

    return fig, stretch_slider


def make_default_export_name(tpf, suffix="custom-lc"):
    """makes the default name to save a custom interact mask"""
    fn = tpf.hdu.filename()
    if fn is None:
        outname = "{}_{}_{}.fits".format(tpf.mission, tpf.targetid, suffix)
    else:
        base = os.path.basename(fn)
        outname = base.rsplit(".fits")[0] + "-{}.fits".format(suffix)
    return outname


def show_interact_widget(
    tpf,
    notebook_url=None,
    lc=None,
    max_cadences=200000,
    aperture_mask="default",
    exported_filename=None,
    transform_func=None,
    ylim_func=None,
    vmin=None,
    vmax=None,
    scale="log",
    cmap="Viridis256",
):
    """Display an interactive Jupyter Notebook widget to inspect the pixel data.

    The widget will show both the lightcurve and pixel data.  The pixel data
    supports pixel selection via Bokeh tap and box select tools in an
    interactive javascript user interface.

    Note: at this time, this feature only works inside an active Jupyter
    Notebook, and tends to be too slow when more than ~30,000 cadences
    are contained in the TPF (e.g. short cadence data).

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Target Pixel File to interact with
    notebook_url: str
        Location of the Jupyter notebook page (default: "localhost:8888")
        When showing Bokeh applications, the Bokeh server must be
        explicitly configured to allow connections originating from
        different URLs. This parameter defaults to the standard notebook
        host and port. If you are running on a different location, you
        will need to supply this value for the application to display
        properly. If no protocol is supplied in the URL, e.g. if it is
        of the form "localhost:8888", then "http" will be used.
        For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
        to the public hostname of your JupyterHub and notebook_url will
        be defined appropriately automatically.
    max_cadences: int
        Raise a RuntimeError if the number of cadences shown is larger than
        this value. This limit helps keep browsers from becoming unresponsive.
    aperture_mask : array-like, 'pipeline', 'threshold', 'default', or 'all'
        A boolean array describing the aperture such that `True` means
        that the pixel will be used.
        If None or 'all' are passed, all pixels will be used.
        If 'pipeline' is passed, the mask suggested by the official pipeline
        will be returned.
        If 'threshold' is passed, all pixels brighter than 3-sigma above
        the median flux will be used.
        If 'default' is passed, 'pipeline' mask will be used when available,
        with 'threshold' as the fallback.
    exported_filename: str
        An optional filename to assign to exported fits files containing
        the custom aperture mask generated by clicking on pixels in interact.
        The default adds a suffix '-custom-aperture-mask.fits' to the
        TargetPixelFile basename.
    transform_func: function
        A function that transforms the lightcurve.  The function takes in a
        LightCurve object as input and returns a LightCurve object as output.
        The function can be complex, such as detrending the lightcurve.  In this
        way, the interactive selection of aperture mask can be evaluated after
        inspection of the transformed lightcurve.  The transform_func is applied
        before saving a fits file.  Default: None (no transform is applied).
    ylim_func: function
        A function that returns ylimits (low, high) given a LightCurve object.
        The default is to return an expanded window around the 10-90th
        percentile of lightcurve flux values.
    scale: str
        Color scale for tpf figure. Default is 'log'
    vmin: int [optional]
        Minimum color scale for tpf figure
    vmax: int [optional]
        Maximum color scale for tpf figure
    cmap: str
        Colormap to use for tpf plot. Default is 'Viridis256'
    """
    if _BOKEH_IMPORT_ERROR is not None:
        log.error(
            "The interact() tool requires the `bokeh` Python package; "
            "you can install bokeh using e.g. `conda install bokeh`."
        )
        raise _BOKEH_IMPORT_ERROR

    notebook_url = finalize_notebook_url(notebook_url)

    aperture_mask = tpf._parse_aperture_mask(aperture_mask)
    if ~aperture_mask.any():
        log.error(
            "No pixels in `aperture_mask`, finding optimum aperture using `tpf.create_threshold_mask`."
        )
        aperture_mask = tpf.create_threshold_mask()
    if ~aperture_mask.any():
        log.error("No pixels in `aperture_mask`, using all pixels.")
        aperture_mask = tpf._parse_aperture_mask("all")

    if exported_filename is None:
        exported_filename = make_default_export_name(tpf)
    try:
        exported_filename = str(exported_filename)
    except:
        log.error("Invalid input filename type for interact()")
        raise
    if ".fits" not in exported_filename.lower():
        exported_filename += ".fits"

    if lc is None:
        lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
        tools = "tap,box_select,wheel_zoom,reset"
    else:
        lc = lc.copy()
        tools = "wheel_zoom,reset"
        aperture_mask = np.zeros(tpf.flux.shape[1:]).astype(bool)
        aperture_mask[0, 0] = True

    lc.meta["APERTURE_MASK"] = aperture_mask

    if transform_func is not None:
        lc = transform_func(lc)

    # Bokeh cannot handle many data points
    # https://github.com/bokeh/bokeh/issues/7490
    n_cadences = len(lc.cadenceno)
    if n_cadences > max_cadences:
        log.error(
            f"Error: interact cannot display more than {max_cadences} "
            "cadences without suffering significant performance issues. "
            "You can limit the number of cadences show using slicing, e.g. "
            "`tpf[0:1000].interact()`. Alternatively, you can override "
            "this limitation by passing the `max_cadences` argument."
        )
    elif n_cadences > 30000:
        log.warning(
            f"Warning: the pixel file contains {n_cadences} cadences. "
            "The performance of interact() is very slow for such a "
            "large number of frames. Consider using slicing, e.g. "
            "`tpf[0:1000].interact()`, to make interact run faster."
        )

    def create_interact_ui(doc):
        # The data source includes metadata for hover-over tooltips
        lc_source = prepare_lightcurve_datasource(lc)
        tpf_source = prepare_tpf_datasource(tpf, aperture_mask)

        # Create the lightcurve figure and its vertical marker
        fig_lc, vertical_line = make_lightcurve_figure_elements(
            lc, lc_source, ylim_func=ylim_func
        )

        # Create the TPF figure and its stretch slider
        pedestal = -np.nanmin(tpf.flux.value) + 1
        if scale == "linear":
            pedestal = 0
        fig_tpf, stretch_slider = make_tpf_figure_elements(
            tpf,
            tpf_source,
            pedestal=pedestal,
            fiducial_frame=0,
            vmin=vmin,
            vmax=vmax,
            scale=scale,
            cmap=cmap,
            tools=tools,
        )

        # Helper lookup table which maps cadence number onto flux array index.
        tpf_index_lookup = {cad: idx for idx, cad in enumerate(tpf.cadenceno)}

        # Interactive slider widgets and buttons to select the cadence number
        cadence_slider = Slider(
            start=np.min(tpf.cadenceno),
            end=np.max(tpf.cadenceno),
            value=np.min(tpf.cadenceno),
            step=1,
            title="Cadence Number",
            width=490,
        )
        r_button = Button(label=">", button_type="default", width=30)
        l_button = Button(label="<", button_type="default", width=30)
        export_button = Button(
            label="Save Lightcurve", button_type="success", width=120
        )
        message_on_save = Div(text=" ", width=600, height=15)

        # Callbacks
        def _create_lightcurve_from_pixels(
            tpf, selected_pixel_indices, transform_func=transform_func
        ):
            """Create the lightcurve from the selected pixel index list"""
            selected_mask = aperture_mask_from_selected_indices(selected_pixel_indices, tpf)
            lc_new = tpf.to_lightcurve(aperture_mask=selected_mask)
            lc_new.meta["APERTURE_MASK"] = selected_mask
            if transform_func is not None:
                lc_transformed = transform_func(lc_new)
                if len(lc_transformed) != len(lc_new):
                    warnings.warn(
                        "Dropping cadences in `transform_func` is not "
                        "yet supported due to fixed time coordinates."
                        "Skipping the transformation...",
                        LightkurveWarning,
                    )
                else:
                    lc_new = lc_transformed
                    lc_new.meta["APERTURE_MASK"] = selected_mask
            return lc_new

        def update_upon_pixel_selection(attr, old, new):
            """Callback to take action when pixels are selected."""
            # Check if a selection was "re-clicked", then de-select
            if (sorted(old) == sorted(new)) & (new != []):
                # Trigger recursion
                tpf_source.selected.indices = new[1:]

            if new != []:
                lc_new = _create_lightcurve_from_pixels(
                    tpf, new, transform_func=transform_func
                )
                lc_source.data["flux"] = lc_new.flux.value

                if ylim_func is None:
                    ylims = get_lightcurve_y_limits(lc_source)
                else:
                    ylims = _to_unitless(ylim_func(lc_new))
                fig_lc.y_range.start = ylims[0]
                fig_lc.y_range.end = ylims[1]
            else:
                lc_source.data["flux"] = lc.flux.value * 0.0
                fig_lc.y_range.start = -1
                fig_lc.y_range.end = 1

            message_on_save.text = " "
            export_button.button_type = "success"

        def update_upon_cadence_change(attr, old, new):
            """Callback to take action when cadence slider changes"""
            if new in tpf.cadenceno:
                frameno = tpf_index_lookup[new]
                fig_tpf.select("tpfimg")[0].data_source.data["image"] = [
                    tpf.flux.value[frameno, :, :] + pedestal
                ]
                vertical_line.update(location=tpf.time.value[frameno])
            else:
                fig_tpf.select("tpfimg")[0].data_source.data["image"] = [
                    tpf.flux.value[0, :, :] * np.NaN
                ]
            lc_source.selected.indices = []

        def go_right_by_one():
            """Step forward in time by a single cadence"""
            existing_value = cadence_slider.value
            if existing_value < np.max(tpf.cadenceno):
                cadence_slider.value = existing_value + 1

        def go_left_by_one():
            """Step back in time by a single cadence"""
            existing_value = cadence_slider.value
            if existing_value > np.min(tpf.cadenceno):
                cadence_slider.value = existing_value - 1

        def save_lightcurve():
            """Save the lightcurve as a fits file with mask as HDU extension"""
            if tpf_source.selected.indices != []:
                lc_new = _create_lightcurve_from_pixels(
                    tpf, tpf_source.selected.indices, transform_func=transform_func
                )
                lc_new.to_fits(
                    exported_filename,
                    overwrite=True,
                    flux_column_name="SAP_FLUX",
                    aperture_mask=lc_new.meta["APERTURE_MASK"].astype(np.int_),
                    SOURCE="lightkurve interact",
                    NOTE="custom mask",
                    MASKNPIX=np.nansum(lc_new.meta["APERTURE_MASK"]),
                )
                if message_on_save.text == " ":
                    text = '<font color="black"><i>Saved file {} </i></font>'
                    message_on_save.text = text.format(exported_filename)
                    export_button.button_type = "success"
                else:
                    text = '<font color="gray"><i>Saved file {} </i></font>'
                    message_on_save.text = text.format(exported_filename)
            else:
                text = (
                    '<font color="gray"><i>No pixels selected, no mask saved</i></font>'
                )
                export_button.button_type = "warning"
                message_on_save.text = text

        def jump_to_lightcurve_position(attr, old, new):
            if new != []:
                cadence_slider.value = lc.cadenceno[new[0]]

        # Map changes to callbacks
        r_button.on_click(go_right_by_one)
        l_button.on_click(go_left_by_one)
        tpf_source.selected.on_change("indices", update_upon_pixel_selection)
        lc_source.selected.on_change("indices", jump_to_lightcurve_position)
        export_button.on_click(save_lightcurve)
        cadence_slider.on_change("value", update_upon_cadence_change)

        # Layout all of the plots
        sp1, sp2, sp3, sp4 = (
            Spacer(width=15),
            Spacer(width=30),
            Spacer(width=80),
            Spacer(width=60),
        )
        widgets_and_figures = layout(
            [fig_lc, fig_tpf],
            [l_button, sp1, r_button, sp2, cadence_slider, sp3, stretch_slider],
            [export_button, sp4, message_on_save],
        )
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)



def show_skyview_widget(tpf, notebook_url=None, aperture_mask="empty",  magnitude_limit=18):
    """skyview

    Parameters
    ----------
    tpf : lightkurve.TargetPixelFile
        Target Pixel File to interact with
    notebook_url: str
        Location of the Jupyter notebook page (default: "localhost:8888")
        When showing Bokeh applications, the Bokeh server must be
        explicitly configured to allow connections originating from
        different URLs. This parameter defaults to the standard notebook
        host and port. If you are running on a different location, you
        will need to supply this value for the application to display
        properly. If no protocol is supplied in the URL, e.g. if it is
        of the form "localhost:8888", then "http" will be used.
        For use with JupyterHub, set the environment variable LK_JUPYTERHUB_EXTERNAL_URL
        to the public hostname of your JupyterHub and notebook_url will
        be defined appropriately automatically.
    aperture_mask : array-like, 'pipeline', 'threshold', 'default', 'background', or 'empty'
        Highlight pixels selected by aperture_mask.
        Default is 'empty': no pixel is highlighted.
    magnitude_limit : float
        A value to limit the results in based on Gaia Gmag. Default, 18.
    """
    if _BOKEH_IMPORT_ERROR is not None:
        log.error(
            "The interact_sky() tool requires the `bokeh` Python package; "
            "you can install bokeh using e.g. `conda install bokeh`."
        )
        raise _BOKEH_IMPORT_ERROR

    notebook_url = finalize_notebook_url(notebook_url)

    # Try to identify the "fiducial frame", for which the TPF WCS is exact
    zp = (tpf.pos_corr1 == 0) & (tpf.pos_corr2 == 0)
    (zp_loc,) = np.where(zp)

    if len(zp_loc) == 1:
        fiducial_frame = zp_loc[0]
    else:
        fiducial_frame = 0

    aperture_mask = tpf._parse_aperture_mask(aperture_mask)
    def create_interact_ui(doc):
        tpf_source = prepare_tpf_datasource(tpf, aperture_mask)

        # The data source includes metadata for hover-over tooltips

        # Create the TPF figure and its stretch slider
        fig_tpf, stretch_slider = make_tpf_figure_elements(
            tpf,
            tpf_source,
            tpf_source_selectable=False,
            fiducial_frame=fiducial_frame,
            width=640,
            height=600,
            tools="tap,box_zoom,wheel_zoom,reset"
        )
        fig_tpf, r, message_selected_target = add_gaia_figure_elements(
            tpf, fig_tpf, magnitude_limit=magnitude_limit
        )

        # Optionally override the default title
        if tpf.mission == "K2":
            fig_tpf.title.text = (
                "Skyview for EPIC {}, K2 Campaign {}, CCD {}.{}".format(
                    tpf.targetid, tpf.campaign, tpf.module, tpf.output
                )
            )
        elif tpf.mission == "Kepler":
            fig_tpf.title.text = (
                "Skyview for KIC {}, Kepler Quarter {}, CCD {}.{}".format(
                    tpf.targetid, tpf.quarter, tpf.module, tpf.output
                )
            )
        elif tpf.mission == "TESS":
            fig_tpf.title.text = "Skyview for TESS {} Sector {}, Camera {}.{}".format(
                tpf.targetid, tpf.sector, tpf.camera, tpf.ccd
            )

        # Layout all of the plots
        widgets_and_figures = layout([fig_tpf, message_selected_target], [stretch_slider])
        doc.add_root(widgets_and_figures)

    output_notebook(verbose=False, hide_banner=True)
    return show(create_interact_ui, notebook_url=notebook_url)
