"""Defines LightCurveFile classes, i.e. files that contain LightCurves."""

from __future__ import division, print_function

import os
import logging
import warnings

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from astropy.io import fits as pyfits

from .utils import (bkjd_to_astropy_time, KeplerQualityFlags, TessQualityFlags,
                    LightkurveWarning)

from . import MPLSTYLE

__all__ = ['KeplerLightCurveFile', 'TessLightCurveFile']

log = logging.getLogger(__name__)


class LightCurveFile(object):
    """Defines a generic class to handle light curve files.

    Parameters
    ----------
    path : str or `astropy.io.fits.HDUList` object
        Local path or remote url of a lightcurve FITS file.
        Also accepts a FITS file object already opened using AstroPy.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, **kwargs):
        if isinstance(path, pyfits.HDUList):
            self.path = None
            self.hdu = path
        else:
            self.path = path
            self.hdu = pyfits.open(self.path, **kwargs)

    def header(self, ext=0):
        """Header of the object at extension `ext`"""
        return self.hdu[ext].header

    @property
    def time(self):
        """Time measurements"""
        return self.hdu[1].data['TIME'][self.quality_mask]

    @property
    def ra(self):
        """Right Ascension of the target."""
        return self.hdu[0].header['RA_OBJ']

    @property
    def dec(self):
        """Declination of the target."""
        return self.hdu[0].header['DEC_OBJ']

    @property
    def SAP_FLUX(self):
        """Returns a LightCurve object for SAP_FLUX"""
        return self.get_lightcurve('SAP_FLUX')

    @property
    def PDCSAP_FLUX(self):
        """Returns a LightCurve object for PDCSAP_FLUX"""
        return self.get_lightcurve('PDCSAP_FLUX')

    @property
    def cadenceno(self):
        """Cadence number"""
        return self.hdu[1].data['CADENCENO'][self.quality_mask]

    @classmethod
    def from_fits(cls, path_or_url, **kwargs):
        """WARNING: THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED VERY SOON.

        Please use `lightkurve.open()` instead.
        """
        warnings.warn('`LightCurveFile.from_fits()` is deprecated and will be '
                      'removed soon, please use `lightkurve.open()` instead.',
                      LightkurveWarning)
        return cls(path_or_url, **kwargs)

    def _flux_types(self):
        """Returns a list of available flux types for this light curve file"""
        types = [n for n in self.hdu[1].data.columns.names if 'FLUX' in n]
        types = [n for n in types if not ('ERR' in n)]
        return types

    def plot(self, flux_types=None, style='lightkurve', **kwargs):
        """Plot all the light curves contained in this light curve file.

        Parameters
        ----------
        flux_types : str or list of str
            List of flux types to plot. Default is to plot all available.
            (For Kepler the default fluxes are 'SAP_FLUX' and 'PDCSAP_FLUX'.
        style : str
            matplotlib.pyplot.style.context, default is 'fast'
        kwargs : dict
            Dictionary of keyword arguments to be passed to
            `KeplerLightCurve.plot()`.
        """
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        with plt.style.context(style):
            if not ('ax' in kwargs):
                fig, ax = plt.subplots(1)
                kwargs['ax'] = ax
            if flux_types is None:
                flux_types = self._flux_types()
            if isinstance(flux_types, str):
                flux_types = [flux_types]
            for idx, ft in enumerate(flux_types):
                lc = self.get_lightcurve(ft)
                kwargs['color'] = np.asarray(mpl.rcParams['axes.prop_cycle'])[idx]['color']
                lc.plot(label=ft, **kwargs)


class KeplerLightCurveFile(LightCurveFile):
    """Defines a class for a given light curve FITS file from NASA's Kepler and
    K2 missions.

    Parameters
    ----------
    path : str
        Local path or remote url of a FITS file in Kepler's lightcurve format.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=1130799`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=1664431`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=2096639`). This mask is not recommended.

        See the :class:`KeplerQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(KeplerLightCurveFile, self).__init__(path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = KeplerQualityFlags.create_quality_mask(
                                quality_array=self.hdu[1].data['SAP_QUALITY'],
                                bitmask=quality_bitmask)
        try:
            self.targetid = self.header()['KEPLERID']
        except KeyError:
            self.targetid = None

    @staticmethod
    def from_archive(target, cadence='long', quarter=None, month=None,
                     campaign=None, quality_bitmask="default", **kwargs):
        """WARNING: THIS FUNCTION IS DEPRECATED AND WILL BE REMOVED VERY SOON.
        Use `lightkurve.search_lightcurvefile()` instead.

        Parameters
        ----------
        target : str or int
            KIC/EPIC ID or object name.
        cadence : str
            'long' or 'short'.
        quarter, campaign : int, list of ints, or 'all'
            Kepler Quarter or K2 Campaign number.
        month : 1, 2, 3, list of int, or 'all'
            For Kepler's prime mission, there are three short-cadence
            LightCurveFile objects for each quarter, each covering one month.
            Hence, if cadence='short' you need to specify month=1, 2, or 3.
        quality_bitmask : str or int
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored (`quality_bitmask=0`).
                * "default": cadences with severe quality issues will be ignored
                  (`quality_bitmask=1130799`).
                * "hard": more conservative choice of flags to ignore
                  (`quality_bitmask=1664431`). This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  (`quality_bitmask=2096639`). This mask is not recommended.

            See the :class:`KeplerQualityFlags` class for details on the bitmasks.
        kwargs : dict
            Keywords arguments passed to `KeplerLightCurveFile`.

        Returns
        -------
        lcf : KeplerLightCurveFile or LightCurveFileCollection
        """
        warnings.warn("`LightCurveFile.from_archive()` is deprecated and will be removed soon, "
                      "please use `lightkurve.search_lightcurvefile()` instead.",
                      LightkurveWarning)

        # Be tolerant if a direct path or url is passed to this function by accident
        if os.path.exists(str(target)) or str(target).startswith('http'):
            log.warning('Warning: from_archive() is not intended to accept a '
                        'direct path, use KeplerLightCurveFile(path) instead.')
            KeplerLightCurveFile(target)
        else:
            from .search import search_lightcurvefile
            sr = search_lightcurvefile(target, cadence=cadence,
                                       quarter=quarter, month=month,
                                       campaign=campaign)
            if len(sr) == 1:
                return sr.download(quality_bitmask=quality_bitmask, **kwargs)
            elif len(sr) > 1:
                return sr.download_all(quality_bitmask=quality_bitmask, **kwargs)
            else:
                raise ValueError("No light curve files found that match the search criteria.")

    def __repr__(self):
        return('KeplerLightCurveFile(ID: {})'.format(self.targetid))

    @property
    def astropy_time(self):
        """Returns an AstroPy Time object for all good-quality cadences."""
        return bkjd_to_astropy_time(bkjd=self.time)

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if flux_type in self._flux_types():
            # We did not import lightcurve at the top to prevent circular imports
            from .lightcurve import KeplerLightCurve
            return KeplerLightCurve(
                time=self.hdu[1].data['TIME'][self.quality_mask],
                time_format='bkjd',
                time_scale='tdb',
                flux=self.hdu[1].data[flux_type][self.quality_mask],
                flux_err=self.hdu[1].data[flux_type + "_ERR"][self.quality_mask],
                centroid_col=self.hdu[1].data[centroid_type + "1"][self.quality_mask],
                centroid_row=self.hdu[1].data[centroid_type + "2"][self.quality_mask],
                quality=self.hdu[1].data['SAP_QUALITY'][self.quality_mask],
                quality_bitmask=self.quality_bitmask,
                channel=self.channel,
                campaign=self.campaign,
                quarter=self.quarter,
                mission=self.mission,
                cadenceno=self.cadenceno,
                targetid=self.targetid,
                label=self.hdu[0].header['OBJECT'],
                ra=self.ra,
                dec=self.dec)
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types))

    @property
    def channel(self):
        """Kepler CCD channel number. ('CHANNEL' header keyword)"""
        return self.header(ext=0)['CHANNEL']

    @property
    def obsmode(self):
        """'short cadence' or 'long cadence'. ('OBSMODE' header keyword)"""
        return self.header()['OBSMODE']

    @property
    def pos_corr1(self):
        """Returns the column position correction."""
        return self.hdu[1].data['POS_CORR1'][self.quality_mask]

    @property
    def pos_corr2(self):
        """Returns the row position correction."""
        return self.hdu[1].data['POS_CORR2'][self.quality_mask]

    @property
    def quarter(self):
        """Kepler quarter number. ('QUARTER' header keyword)"""
        try:
            return self.header(ext=0)['QUARTER']
        except KeyError:
            return None

    @property
    def campaign(self):
        """K2 Campaign number. ('CAMPAIGN' header keyword)"""
        try:
            return self.header(ext=0)['CAMPAIGN']
        except KeyError:
            return None

    @property
    def mission(self):
        """'Kepler' or 'K2'. ('MISSION' header keyword)"""
        try:
            return self.header(ext=0)['MISSION']
        except KeyError:
            return None

    def compute_cotrended_lightcurve(self, cbvs=[1, 2], **kwargs):
        """Returns a LightCurve object after cotrending the SAP_FLUX
        against the cotrending basis vectors.

        Parameters
        ----------
        cbvs : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
        kwargs : dict
            Dictionary of keyword arguments to be passed to
            KeplerCBVCorrector.correct.

        Returns
        -------
        lc : LightCurve object
            CBV flux-corrected lightcurve.
        """
        from .correctors import KeplerCBVCorrector
        return KeplerCBVCorrector(self).correct(cbvs=cbvs, **kwargs)


class TessLightCurveFile(LightCurveFile):
    """Defines a class for a given light curve FITS file from NASA's TESS
    mission.

    Parameters
    ----------
    path : str
        Local path or remote url of a FITS file in TESS's lightcurve format.
    quality_bitmask : str or int
        Bitmask (integer) which identifies the quality flag bitmask that should
        be used to mask out bad cadences. If a string is passed, it has the
        following meaning:

            * "none": no cadences will be ignored (`quality_bitmask=0`).
            * "default": cadences with severe quality issues will be ignored
              (`quality_bitmask=1130799`).
            * "hard": more conservative choice of flags to ignore
              (`quality_bitmask=1664431`). This is known to remove good data.
            * "hardest": removes all data that has been flagged
              (`quality_bitmask=2096639`). This mask is not recommended.

        See the :class:`TessQualityFlags` class for details on the bitmasks.
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, quality_bitmask='default', **kwargs):
        super(TessLightCurveFile, self).__init__(path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = TessQualityFlags.create_quality_mask(
                                quality_array=self.hdu[1].data['QUALITY'],
                                bitmask=quality_bitmask)
        # Early TESS releases had cadences with time=NaN (i.e. missing data)
        # which were not flagged by a QUALITY flag yet; the line below prevents
        # these cadences from being used. They would break most methods!
        self.quality_mask &= np.isfinite(self.hdu[1].data['TIME'])
        try:
            self.targetid = self.header()['TICID']
        except KeyError:
            self.targetid = None

    def __repr__(self):
        return('TessLightCurveFile(TICID: {})'.format(self.targetid))

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if flux_type in self._flux_types():
            # We did not import TessLightCurve at the top to prevent circular imports
            from .lightcurve import TessLightCurve
            return TessLightCurve(
                time=self.hdu[1].data['TIME'][self.quality_mask],
                time_format='btjd',
                time_scale='tdb',
                flux=self.hdu[1].data[flux_type][self.quality_mask],
                flux_err=self.hdu[1].data[flux_type + "_ERR"][self.quality_mask],
                centroid_col=self.hdu[1].data[centroid_type + "1"][self.quality_mask],
                centroid_row=self.hdu[1].data[centroid_type + "2"][self.quality_mask],
                quality=self.hdu[1].data['QUALITY'][self.quality_mask],
                quality_bitmask=self.quality_bitmask,
                cadenceno=self.cadenceno,
                targetid=self.targetid,
                label=self.hdu[0].header['OBJECT'])
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types))
