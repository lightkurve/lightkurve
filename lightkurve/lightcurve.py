import copy
import numpy as np
import oktopus
from scipy import linalg, signal, interpolate
from astropy.io import fits as pyfits
from astropy.stats import sigma_clip
from astropy.table import Table
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from .utils import running_mean, channel_to_module_output, KeplerQualityFlags


__all__ = ['LightCurve', 'KeplerLightCurve', 'KeplerLightCurveFile',
           'KeplerCBVCorrector', 'SPLDCorrector', 'SFFCorrector',
           'box_period_search']


class LightCurve(object):
    """
    Implements a simple class for a generic light curve.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    meta : dict
        Free-form metadata associated with the LightCurve.
    """

    def __init__(self, time, flux, flux_err=None, meta={}):
        self.time = np.asarray(time)
        self.flux = np.asarray(flux)
        if flux_err is not None:
            self.flux_err = np.asarray(flux_err)
        else:
            self.flux_err = np.nan * np.ones_like(self.time)
        self.meta = meta

    def stitch(self, *others):
        """
        Stitches LightCurve objects.

        Parameters
        ----------
        *others : LightCurve objects
            Light curves to be stitched.

        Returns
        -------
        stitched_lc : LightCurve object
            Stitched light curve.
        """
        time = self.time
        flux = self.flux
        flux_err = self.flux_err

        for i in range(len(others)):
            time = np.append(time, others[i].time)
            flux = np.append(flux, others[i].flux)
            flux_err = np.append(flux_err, others[i].flux_err)

        return LightCurve(time=time, flux=flux, flux_err=flux_err)

    def flatten(self, window_length=101, polyorder=3, return_trend=False, **kwargs):
        """
        Removes low frequency trend using scipy's Savitzky-Golay filter.

        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e. the number of coefficients).
            ``window_length`` must be a positive odd integer.
        polyorder : int
            The order of the polynomial used to fit the samples. ``polyorder``
            must be less than window_length.
        return_trend : bool
            If `True`, the method will return a tuple of two elements
            (flattened_lc, trend_lc) where trend_lc is the removed trend.
        **kwargs : dict
            Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.

        Returns
        -------
        flatten_lc : LightCurve object
            Flattened lightcurve.
        If `return_trend` is `True`, the method will also return:
        trend_lc : LightCurve object
            Trend in the lightcurve data
        """
        lc_clean = self.remove_nans()  # The SG filter does not allow NaNs
        trend_signal = signal.savgol_filter(x=lc_clean.flux,
                                            window_length=window_length,
                                            polyorder=polyorder, **kwargs)
        flatten_lc = copy.copy(lc_clean)
        flatten_lc.flux = lc_clean.flux / trend_signal
        if flatten_lc.flux_err is not None:
            flatten_lc.flux_err = lc_clean.flux_err / trend_signal

        if return_trend:
            trend_lc = copy.copy(lc_clean)
            trend_lc.flux = trend_signal
            return flatten_lc, trend_lc
        return flatten_lc

    def fold(self, period, phase=0.):
        """Folds the lightcurve at a specified ``period`` and ``phase``.

        This method returns a new ``LightCurve`` object in which the time
        values range between -0.5 to +0.5.  Data points which occur exactly
        at ``phase`` or an integer multiple of `phase + n*period` have time
        value 0.0.

        Parameters
        ----------
        period : float
            The period upon which to fold.
        phase : float, optional
            Time reference point.

        Returns
        -------
        folded_lightcurve : LightCurve object
            A new ``LightCurve`` in which the data are folded and sorted by
            phase.
        """
        fold_time = ((self.time - phase + 0.5 * period) / period) % 1 - 0.5
        sorted_args = np.argsort(fold_time)
        if self.flux_err is None:
            return LightCurve(fold_time[sorted_args], self.flux[sorted_args])
        return LightCurve(fold_time[sorted_args], self.flux[sorted_args], flux_err=self.flux_err[sorted_args])

    def normalize(self):
        """Returns a normalized version of the lightcurve.

        The normalized lightcurve is obtained by dividing `flux` and `flux_err`
        by the median flux.

        Returns
        -------
        normalized_lightcurve : LightCurve object
            A new ``LightCurve`` in which `flux` and `flux_err` are divided
            by the median.
        """
        lc = copy.copy(self)
        if lc.flux_err is not None:
            lc.flux_err = lc.flux_err / np.nanmedian(lc.flux)
        lc.flux = lc.flux / np.nanmedian(lc.flux)
        return lc

    def remove_nans(self):
        """Removes cadences where the flux is NaN.

        Returns
        -------
        clean_lightcurve : LightCurve object
            A new ``LightCurve`` from which NaNs fluxes have been removed.
        """
        lc = copy.copy(self)
        nan_mask = np.isnan(lc.flux)
        lc.time = self.time[~nan_mask]
        lc.flux = self.flux[~nan_mask]
        if lc.flux_err is not None:
            lc.flux_err = self.flux_err[~nan_mask]
        return lc

    def remove_outliers(self, sigma=5., return_mask=False, **kwargs):
        """Removes outlier flux values using sigma-clipping.

        This method returns a new LightCurve object from which flux values
        are removed if they are separated from the mean flux by `sigma` times
        the standard deviation.

        Parameters
        ----------
        sigma : float
            The number of standard deviations to use for clipping outliers.
            Defaults to 5.
        return_mask : bool
            Whether or not to return the mask indicating which data points
            were removed. Entries marked as `True` are considered outliers.
        **kwargs : dict
            Dictionary of arguments to be passed to `astropy.stats.sigma_clip`.

        Returns
        -------
        clean_lightcurve : LightCurve object
            A new ``LightCurve`` in which outliers have been removed.
        """
        new_lc = copy.copy(self)
        outlier_mask = sigma_clip(data=new_lc.flux, sigma=sigma, **kwargs).mask
        new_lc.time = self.time[~outlier_mask]
        new_lc.flux = self.flux[~outlier_mask]
        if new_lc.flux_err is not None:
            new_lc.flux_err = self.flux_err[~outlier_mask]

        if return_mask:
            return new_lc, outlier_mask
        return new_lc

    def bin(self, binsize=13, method='mean'):
        """Bins a lightcurve using a function defined by `method`
        on blocks of samples of size `binsize`.

        Parameters
        ----------
        binsize : int
            Number of cadences to include in every bin.
        method: str, one of 'mean' or 'median'
            The summary statistic to return for each bin. Default: 'mean'.

        Returns
        -------
        binned_lc : LightCurve object
            Binned lightcurve.

        Notes
        -----
        - If the ratio between the lightcurve length and the binsize is not
          a whole number, then the remainder of the data points will be
          ignored.
        - If the original lightcurve contains flux uncertainties (flux_err),
          the binned lightcurve will report the root-mean-square error.
          If no uncertainties are included, the binned curve will return the
          standard deviation of the data.
        """
        available_methods = ['mean', 'median']
        if method not in available_methods:
            raise ValueError("method must be one of: {}".format(available_methods))
        methodf = np.__dict__['nan' + method]

        n_bins = self.flux.size // binsize
        binned_lc = copy.copy(self)
        binned_lc.time = np.array([methodf(a) for a in np.array_split(self.time, n_bins)])
        binned_lc.flux = np.array([methodf(a) for a in np.array_split(self.flux, n_bins)])

        if self.flux_err is not None:
            # root-mean-square error
            binned_lc.flux_err = np.array(
                                    [np.sqrt(np.nansum(a**2))
                                     for a in np.array_split(self.flux_err, n_bins)]
                                 ) / binsize
        else:
            # compute the standard deviation from the data
            binned_lc.flux_err = np.array([np.nanstd(a)
                                           for a in np.array_split(self.flux, n_bins)])

        return binned_lc

    def cdpp(self, transit_duration=13, savgol_window=101, savgol_polyorder=2,
             sigma_clip=5.):
        """Estimate the CDPP noise metric using the Savitzky-Golay (SG) method.

        A common estimate of the noise in a lightcurve is the scatter that
        remains after all long term trends have been removed. This is the idea
        behind the Combined Differential Photometric Precision (CDPP) metric.
        The official Kepler Pipeline computes this metric using a wavelet-based
        algorithm to calculate the signal-to-noise of the specific waveform of
        transits of various durations. In this implementation, we use the
        simpler "sgCDPP proxy algorithm" discussed by Gilliland et al
        (2011ApJS..197....6G) and Van Cleve et al (2016PASP..128g5002V).

        The steps of this algorithm are:
            1. Remove low frequency signals using a Savitzky-Golay filter with
               window length `savgol_window` and polynomial order `savgol_polyorder`.
            2. Remove outliers by rejecting data points which are separated from
               the mean by `sigma_clip` times the standard deviation.
            3. Compute the standard deviation of a running mean with
               a configurable window length equal to `transit_duration`.

        We use a running mean (as opposed to block averaging) to strongly
        attenuate the signal above 1/transit_duration whilst retaining
        the original frequency sampling.  Block averaging would set the Nyquist
        limit to 1/transit_duration.

        Parameters
        ----------
        transit_duration : int, optional
            The transit duration in cadences. This is the length of the window
            used to compute the running mean. The default is 13, which
            corresponds to a 6.5 hour transit in data sampled at 30-min cadence.
        savgol_window : int, optional
            Width of Savitsky-Golay filter in cadences (odd number).
            Default value 101 (2.0 days in Kepler Long Cadence mode).
        savgol_polyorder : int, optional
            Polynomial order of the Savitsky-Golay filter.
            The recommended value is 2.
        sigma_clip : float, optional
            The number of standard deviations to use for clipping outliers.
            The default is 5.

        Returns
        -------
        cdpp : float
            Savitzky-Golay CDPP noise metric in units parts-per-million (ppm).

        Notes
        -----
        This implementation is adapted from the Matlab version used by
        Jeff van Cleve but lacks the normalization factor used there:
        svn+ssh://murzim/repo/so/trunk/Develop/jvc/common/compute_SG_noise.m
        """
        if not isinstance(transit_duration, int):
            raise TypeError("transit_duration must be an integer")
        detrended_lc = self.flatten(window_length=savgol_window,
                                    polyorder=savgol_polyorder)
        cleaned_lc = detrended_lc.remove_outliers(sigma=sigma_clip)
        mean = running_mean(data=cleaned_lc.flux, window_size=transit_duration)
        cdpp_ppm = np.std(mean) * 1e6
        return cdpp_ppm

    def plot(self, ax=None, normalize=True, xlabel='Time - 2454833 (days)',
             ylabel='Normalized Flux', title=None, color='#363636', linestyle="",
             fill=False, grid=True, **kwargs):
        """Plots the light curve.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        color: str
            Color to plot flux points
        fill: bool
            Shade the region between 0 and flux
        grid: bool
            Add a grid to the plot
        **kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if ax is None:
            fig, ax = plt.subplots(1)
        if normalize:
            normalized_lc = self.normalize()
            flux, flux_err = normalized_lc.flux, normalized_lc.flux_err
        else:
            flux, flux_err = self.flux, self.flux_err
        if flux_err is None:
            ax.plot(self.time, flux, marker='o', color=color,
                    linestyle=linestyle, **kwargs)
        else:
            ax.errorbar(self.time, flux, flux_err, color=color,
                        linestyle=linestyle, **kwargs)
        if fill:
            ax.fill(self.time, flux, fc='#a8a7a7', linewidth=0.0, alpha=0.3)
        if grid:
            ax.grid(alpha=0.3)
        if 'label' in kwargs:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel, {'color': 'k'})
        ax.set_ylabel(ylabel, {'color': 'k'})
        return ax

    def to_table(self):
        """Export the LightCurve as an AstroPy Table.

        Returns
        -------
        table : `astropy.table.Table` object
            An AstroPy Table with columns 'time', 'flux', and 'flux_err'.
        """
        return Table(data=(self.time, self.flux, self.flux_err),
                     names=('time', 'flux', 'flux_err'),
                     meta=self.meta)

    def to_pandas(self):
        """Export the LightCurve as a Pandas DataFrame.

        Returns
        -------
        dataframe : `pandas.DataFrame` object
            A dataframe indexed by `time` and containing the columns `flux`
            and `flux_err`.
        """
        try:
            import pandas as pd
        # lightkurve does not require pandas, so check for import success.
        except ImportError:
            raise ImportError("You need to install pandas to use the "
                              "LightCurve.to_pandas() method.")
        df = pd.DataFrame(data={'flux': self.flux, 'flux_err': self.flux_err},
                          index=self.time,
                          columns=['flux', 'flux_err'])
        df.index.name = 'time'
        df.meta = self.meta
        return df

    def to_csv(self, path_or_buf=None, **kwargs):
        """Writes the LightCurve to a csv file.

        Parameters
        ----------
        path_or_buf : string or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.
        **kwargs : dict
            Dictionary of arguments to be passed to `pandas.DataFrame.to_csv()`.

        Returns
        -------
        csv : str or None
            Returns a csv-formatted string if `path_or_buf=None`,
            returns None otherwise.
        """
        return self.to_pandas().to_csv(path_or_buf=path_or_buf, **kwargs)


class KeplerLightCurve(LightCurve):
    """Defines a light curve class for NASA's Kepler and K2 missions.

    Attributes
    ----------
    time : array-like
        Time measurements
    flux : array-like
        Data flux for every time point
    flux_err : array-like
        Uncertainty on each flux data point
    centroid_col, centroid_row : array-like, array-like
        Centroid column and row coordinates as a function of time
    quality : array-like
        Array indicating the quality of each data point
    quality_bitmask : int
        Bitmask specifying quality flags of cadences that should be ignored
    channel : int
        Channel number
    campaign : int
        Campaign number
    quarter : int
        Quarter number
    mission : str
        Mission name
    cadenceno : array-like
        Cadence numbers corresponding to every time measurement
    keplerid : int
        Kepler ID number
    """

    def __init__(self, time, flux, flux_err=None, centroid_col=None,
                 centroid_row=None, quality=None, quality_bitmask=None,
                 channel=None, campaign=None, quarter=None, mission=None,
                 cadenceno=None, keplerid=None):
        super(KeplerLightCurve, self).__init__(time, flux, flux_err)
        self.centroid_col = centroid_col
        self.centroid_row = centroid_row
        self.quality = quality
        self.quality_bitmask = quality_bitmask
        self.channel = channel
        self.campaign = campaign
        self.quarter = quarter
        self.mission = mission
        self.cadenceno = cadenceno
        self.keplerid = keplerid

    def correct(self, method='sff', **kwargs):
        """Corrects a lightcurve for motion-dependent systematic errors.

        Parameters
        ----------
        method : str
            Method used to correct the lightcurve.
            Right now only 'sff' (Vanderburg's Self-Flat Fielding) is supported.
        kwargs : dict
            Dictionary of keyword arguments to be passed to the function
            defined by `method`.

        Returns
        -------
        new_lc : KeplerLightCurve object
            Corrected lightcurve
        """
        if method == 'sff':
            self.corrector = SFFCorrector()
            corrected_lc = self.corrector.correct(time=self.time, flux=self.flux,
                                                  centroid_col=self.centroid_col,
                                                  centroid_row=self.centroid_row,
                                                  **kwargs)
        else:
            raise ValueError("method {} is not available.".format(method))
        new_lc = copy.copy(self)
        new_lc.flux = corrected_lc.flux
        return new_lc

    def to_fits(self):
        raise NotImplementedError()


class KeplerLightCurveFile(object):
    """Defines a class for a given light curve FITS file from NASA's Kepler and
    K2 missions.

    Attributes
    ----------
    path : str
        Directory path or url to a lightcurve FITS file.
    quality_bitmask : str or int
        Bitmask specifying quality flags of cadences that should be ignored.
        If a string is passed, it has the following meaning:

            * default: recommended quality mask
            * hard: removes more flags, known to remove good data
            * hardest: removes all data that has been flagged
    kwargs : dict
        Keyword arguments to be passed to astropy.io.fits.open.
    """
    def __init__(self, path, quality_bitmask=KeplerQualityFlags.DEFAULT_BITMASK,
                 **kwargs):
        self.path = path
        self.hdu = pyfits.open(self.path, **kwargs)
        self.quality_bitmask = quality_bitmask
        self.quality_mask = self._quality_mask(quality_bitmask)

    def get_lightcurve(self, flux_type, centroid_type='MOM_CENTR'):
        if flux_type in self._flux_types():
            return KeplerLightCurve(self.hdu[1].data['TIME'][self.quality_mask],
                                    self.hdu[1].data[flux_type][self.quality_mask],
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
                                    keplerid=self.hdu[0].header['KEPLERID'])
        else:
            raise KeyError("{} is not a valid flux type. Available types are: {}".
                           format(flux_type, self._flux_types))

    def _quality_mask(self, bitmask):
        """Returns a boolean mask which flags all good-quality cadences.

        Parameters
        ----------
        bitmask : str or int
            Bitmask. See ref. [1], table 2-3.

        Returns
        -------
        boolean_mask : array of bool
            Boolean array in which `True` means the data is of good quality.
        """
        if bitmask is None:
            return np.ones(len(self.hdu[1].data['TIME']), dtype=bool)
        elif isinstance(bitmask, str):
            bitmask = KeplerQualityFlags.OPTIONS[bitmask]
        return (self.hdu[1].data['SAP_QUALITY'] & bitmask) == 0

    @property
    def SAP_FLUX(self):
        """Returns a KeplerLightCurve object for SAP_FLUX"""
        return self.get_lightcurve('SAP_FLUX')

    @property
    def PDCSAP_FLUX(self):
        """Returns a KeplerLightCurve object for PDCSAP_FLUX"""
        return self.get_lightcurve('PDCSAP_FLUX')

    @property
    def time(self):
        """Time measurements"""
        return self.hdu[1].data['TIME'][self.quality_mask]

    @property
    def cadenceno(self):
        """Cadence number"""
        return self.hdu[1].data['CADENCENO'][self.quality_mask]

    @property
    def channel(self):
        """Channel number"""
        return self.header(ext=0)['CHANNEL']

    @property
    def quarter(self):
        """Quarter number"""
        try:
            return self.header(ext=0)['QUARTER']
        except KeyError:
            return None

    @property
    def campaign(self):
        """Campaign number"""
        try:
            return self.header(ext=0)['CAMPAIGN']
        except KeyError:
            return None

    @property
    def mission(self):
        """Mission name"""
        return self.header(ext=0)['MISSION']

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
        return KeplerCBVCorrector(self).correct(cbvs=cbvs, **kwargs)

    def header(self, ext=0):
        """Header of the object at extension `ext`"""
        return self.hdu[ext].header

    def _flux_types(self):
        """Returns a list of available flux types for this light curve file"""
        types = [n for n in self.hdu[1].data.columns.names if 'FLUX' in n]
        types = [n for n in types if not ('ERR' in n)]
        return types

    def plot(self, plottype=None, **kwargs):
        """Plot all the flux types in a light curve.

        Parameters
        ----------
        plottype : str or list of str
            List of FLUX types to plot. Default is to plot all available.
        """
        if not ('ax' in kwargs):
            fig, ax = plt.subplots(1)
            kwargs['ax'] = ax
        if not ('title' in kwargs):
            kwargs['title'] = 'KeplerID: {}'.format(self.SAP_FLUX.keplerid)
        if plottype is None:
            plottype = self._flux_types()
        if isinstance(plottype, str):
            plottype = [plottype]
        for idx, pl in enumerate(plottype):
            lc = self.get_lightcurve(pl)
            kwargs['color'] = 'C{}'.format(idx)
            lc.plot(label=pl, **kwargs)


class SFFCorrector(object):
    """Implements the Self-Flat-Fielding (SFF) systematics removal method.

    This method is described in detail by Vanderburg and Johnson (2014).
    Briefly, the algorithm implemented in this class can be described
    as follows

       (1) Rotate the centroid measurements onto the subspace spanned by the
           eigenvectors of the centroid covariance matrix
       (2) Fit a polynomial to the rotated centroids
       (3) Compute the arclength of such polynomial
       (4) Fit a BSpline of the raw flux as a function of time
       (5) Normalize the raw flux by the fitted BSpline computed in step (4)
       (6) Bin and interpolate the normalized flux as function of the arclength
       (7) Divide the raw flux by the piecewise linear interpolation done in step [(6)
       (8) Set raw flux as the flux computed in step (7) and repeat
    """

    def __init__(self):
        pass

    def correct(self, time, flux, centroid_col, centroid_row,
                polyorder=5, niters=3, bins=15, windows=1, sigma_1=3.,
                sigma_2=5.):
        """Returns a systematics-corrected LightCurve.

        Parameters
        ----------
        time : array-like
            Time measurements
        flux : array-like
            Data flux for every time point
        centroid_col, centroid_row : array-like, array-like
            Centroid column and row coordinates as a function of time
        polyorder : int
            Degree of the polynomial which will be used to fit one
            centroid as a function of the other.
        niters : int
            Number of iterations of the aforementioned algorithm.
        bins : int
            Number of bins to be used in step (6) to create the
            piece-wise interpolation of arclength vs flux correction.
        windows : int
            Number of windows to subdivide the data.  The SFF algorithm
            is ran independently in each window.
        sigma_1, sigma_2 : float, float
            Sigma values which will be used to reject outliers
            in steps (6) and (2), respectivelly.

        Returns
        -------
        corrected_lightcurve : LightCurve object
            Returns a corrected lightcurve object.
        """
        timecopy = time
        time = np.array_split(time, windows)
        flux = np.array_split(flux, windows)
        centroid_col = np.array_split(centroid_col, windows)
        centroid_row = np.array_split(centroid_row, windows)

        flux_hat = np.array([])
        # The SFF algorithm is going to be run on each window independently
        for i in tqdm(range(windows)):
            # To make it easier (and more numerically stable) to fit a
            # characteristic polynomial that describes the spacecraft motion,
            # we rotate the centroids to a new coordinate frame in which
            # the dominant direction of motion is aligned with the x-axis.
            self.rot_col, self.rot_row = self.rotate_centroids(centroid_col[i],
                                                               centroid_row[i])
            # Next, we fit the motion polynomial after removing outliers
            self.outlier_cent = sigma_clip(data=self.rot_col,
                                           sigma=sigma_2).mask
            coeffs = np.polyfit(self.rot_row[~self.outlier_cent],
                                self.rot_col[~self.outlier_cent], polyorder)
            self.poly = np.poly1d(coeffs)
            self.polyprime = np.poly1d(coeffs).deriv()

            # Compute the arclength s.  It is the length of the polynomial
            # (fitted above) that describes the typical motion.
            x = np.linspace(np.min(self.rot_row[~self.outlier_cent]),
                            np.max(self.rot_row[~self.outlier_cent]), 10000)
            self.s = np.array([self.arclength(x1=xp, x=x) for xp in self.rot_row])

            # Next, we find and apply the correction iteratively
            for n in range(niters):
                # First, fit a spline to capture the long-term varation
                # We don't want to fit the long-term trend because we know
                # that the K2 motion noise is a high-frequency effect.
                self.bspline = self.fit_bspline(time[i], flux[i])
                # Remove the long-term variation by dividing the flux by the spline
                self.normflux = flux[i] / self.bspline(time[i] - time[i][0])
                # Bin and interpolate normalized flux to capture the dependency
                # of the flux as a function of arclength
                self.interp = self.bin_and_interpolate(self.s, self.normflux, bins,
                                                       sigma=sigma_1)
                # Correct the raw flux
                corrected_flux = self.normflux / self.interp(self.s)
                flux[i] = corrected_flux

            flux_hat = np.append(flux_hat, flux[i])

        return LightCurve(time=timecopy, flux=flux_hat)

    def rotate_centroids(self, centroid_col, centroid_row):
        """Rotate the coordinate frame of the (col, row) centroids to a new (x,y)
        frame in which the dominant motion of the spacecraft is aligned with
        the x axis.  This makes it easier to fit a characteristic polynomial
        that describes the motion."""
        centroids = np.array([centroid_col, centroid_row])
        _, eig_vecs = linalg.eigh(np.cov(centroids))
        return np.dot(eig_vecs, centroids)

    def _plot_rotated_centroids(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.rot_row[~self.outlier_cent], self.rot_col[~self.outlier_cent],
                'ko', markersize=3)
        ax.plot(self.rot_row[~self.outlier_cent], self.rot_col[~self.outlier_cent],
                'bo', markersize=2)
        ax.plot(self.rot_row[self.outlier_cent], self.rot_col[self.outlier_cent],
                'ko', markersize=3)
        ax.plot(self.rot_row[self.outlier_cent], self.rot_col[self.outlier_cent],
                'ro', markersize=2)
        x = np.linspace(min(self.rot_row), max(self.rot_row), 200)
        ax.plot(x, self.poly(x), '--')
        plt.xlabel("Rotated row centroid")
        plt.ylabel("Rotated column centroid")
        return ax

    def _plot_normflux_arclength(self):
        idx = np.argsort(self.s)
        s_srtd = self.s[idx]
        normflux_srtd = self.normflux[idx]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(s_srtd[~self.outlier_mask], normflux_srtd[~self.outlier_mask],
                'ko', markersize=3)
        ax.plot(s_srtd[~self.outlier_mask], normflux_srtd[~self.outlier_mask],
                'bo', markersize=2)
        ax.plot(s_srtd[self.outlier_mask], normflux_srtd[self.outlier_mask],
                'ko', markersize=3)
        ax.plot(s_srtd[self.outlier_mask], normflux_srtd[self.outlier_mask],
                'ro', markersize=2)
        ax.plot(s_srtd, self.interp(s_srtd), '--')
        plt.xlabel(r"Arclength $(s)$")
        plt.ylabel(r"Flux $(e^{-}s^{-1})$")
        return ax

    def arclength(self, x1, x):
        """Compute the arclength of the polynomial used to fit the centroid
        measurements.

        Parameters
        ----------
        x1 : float
            Upper limit of the integration domain.
        x : ndarray
            Domain at which the arclength integrand is defined.

        Returns
        -------
        arclength : float
            Result of the arclength integral from x[0] to x1.
        """
        mask = x < x1
        return np.trapz(y=np.sqrt(1 + self.polyprime(x[mask]) ** 2), x=x[mask])

    def fit_bspline(self, time, flux, s=0):
        """s describes the "smoothness" of the spline"""
        time = time - time[0]
        knots = np.arange(0, time[-1], 1.5)
        t, c, k = interpolate.splrep(time, flux, t=knots[1:], s=s, task=-1)
        return interpolate.BSpline(t, c, k)

    def bin_and_interpolate(self, s, normflux, bins, sigma):
        idx = np.argsort(s)
        s_srtd = s[idx]
        normflux_srtd = normflux[idx]

        self.outlier_mask = sigma_clip(data=normflux_srtd, sigma=sigma).mask
        normflux_srtd = normflux_srtd[~self.outlier_mask]
        s_srtd = s_srtd[~self.outlier_mask]

        knots = np.array([np.min(s_srtd)]
                         + [np.median(split) for split in np.array_split(s_srtd, bins)]
                         + [np.max(s_srtd)])
        bin_means = np.array([normflux_srtd[0]]
                             + [np.mean(split) for split in np.array_split(normflux_srtd, bins)]
                             + [normflux_srtd[-1]])
        return interpolate.interp1d(knots, bin_means, bounds_error=False,
                                    fill_value='extrapolate')

    def breakpoints(self, campaign):
        """Return a break point as a function of the campaign number.

        The intention of this function is to implement a smart way to determine
        the boundaries of the windows on which the SFF algorithm is applied
        independently. However, this is not implemented yet in this version.
        """
        raise NotImplementedError()


class KeplerCBVCorrector(object):
    r"""Remove systematic trends from Kepler light curves by fitting
    cotrending basis vectors.

    .. math::

        \arg \min_{\bm{\theta} \in \Theta} \sum_{t}|f_{SAP}(t) - \sum_{j=1}^{n}\theta_j v_{j}(t)|^p, p>0, p \in \mathbb{R}

    Attributes
    ----------
    lc_file : KeplerLightCurveFile object or str
        An instance from KeplerLightCurveFile or a path for the .fits
        file of a NASA's Kepler/K2 light curve.
    likelihood : oktopus.Likelihood subclass
        A class that describes a cost function.
        The default is :class:`oktopus.LaplacianLikelihood`, which is tantamount
        to the L1 norm.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lightkurve import KeplerCBVCorrector, KeplerLightCurveFile
    >>> fn = ("https://archive.stsci.edu/missions/kepler/lightcurves/"
    ...       "0084/008462852/kplr008462852-2011073133259_llc.fits") # doctest: +SKIP
    >>> cbv = KeplerCBVCorrector(fn) # doctest: +SKIP
    Downloading https://archive.stsci.edu/missions/kepler/lightcurves/0084/008462852/kplr008462852-2011073133259_llc.fits [Done]
    >>> cbv_lc = cbv.correct() # doctest: +SKIP
    Downloading http://archive.stsci.edu/missions/kepler/cbv/kplr2011073133259-q08-d25_lcbv.fits [Done]
    >>> sap_lc = KeplerLightCurveFile(fn).SAP_FLUX # doctest: +SKIP
    >>> plt.plot(sap_lc.time, sap_lc.flux, 'x', markersize=1, label='SAP_FLUX') # doctest: +SKIP
    >>> plt.plot(cbv_lc.time, cbv_lc.flux, 'o', markersize=1, label='CBV_FLUX') # doctest: +SKIP
    >>> plt.legend() # doctest: +SKIP
    """

    def __init__(self, lc_file, likelihood=oktopus.LaplacianLikelihood,
                 prior=oktopus.LaplacianPrior):
        self.lc_file = lc_file
        self.likelihood = likelihood
        self.prior = prior
        self._ncbvs = 16 # number of cbvs for Kepler/K2

        if self.lc_file.mission == 'Kepler':
            self.cbv_base_url = "http://archive.stsci.edu/missions/kepler/cbv/"
        elif self.lc_file.mission == 'K2':
            self.cbv_base_url = "http://archive.stsci.edu/missions/k2/cbv/"

    @property
    def lc_file(self):
        return self._lc_file

    @lc_file.setter
    def lc_file(self, value):
        # this enables `lc_file` to be either a string
        # or an object from KeplerLightCurveFile
        if isinstance(value, str):
            self._lc_file = KeplerLightCurveFile(value)
        elif isinstance(value, KeplerLightCurveFile):
            self._lc_file = value
        else:
            raise ValueError("lc_file must be either a string or a"
                             " KeplerLightCurveFile instance, got {}.".format(value))

    @property
    def coeffs(self):
        """
        Returns the fitted coefficients.
        """
        return self._coeffs

    @property
    def opt_result(self):
        """
        Returns the result of the optimization process.
        """
        return self._opt_result

    def correct(self, cbvs=[1, 2], method='powell', options={}):
        """
        Correct the SAP_FLUX by fitting a number of cotrending basis vectors
        `cbvs`.

        Parameters
        ----------
        cbvs : list of ints
            The list of cotrending basis vectors to fit to the data. For example,
            [1, 2] will fit the first two basis vectors.
        method : str
            Numerical optimization method. See scipy.optimize.minimize for the
            full list of methods.
        options : dict
            Dictionary of options to be passed to scipy.optimize.minimize.
        """
        module, output = channel_to_module_output(self.lc_file.channel)
        cbv_file = pyfits.open(self.get_cbv_url())
        cbv_data = cbv_file['MODOUT_{0}_{1}'.format(module, output)].data

        cbv_array = []
        for i in cbvs:
            cbv_array.append(cbv_data.field('VECTOR_{}'.format(i))[self.lc_file.quality_mask])
        cbv_array = np.asarray(cbv_array)

        sap_lc = self.lc_file.SAP_FLUX
        median_sap_flux = np.nanmedian(sap_lc.flux)
        norm_sap_flux = sap_lc.flux / median_sap_flux - 1
        norm_err_sap_flux = sap_lc.flux_err / median_sap_flux

        def mean_model(*theta):
            coeffs = np.asarray(theta)
            return np.dot(coeffs, cbv_array)

        prior = self.prior(mean=np.zeros(len(cbvs)), var=16.)
        likelihood = self.likelihood(data=norm_sap_flux, mean=mean_model,
                                     var=norm_err_sap_flux)
        x0 = likelihood.fit(x0=prior.mean, method=method, options=options).x
        posterior = oktopus.Posterior(likelihood=likelihood, prior=prior)

        self._opt_result = posterior.fit(x0=x0, method=method,
                                         options=options)
        self._coeffs = self._opt_result.x
        flux_hat = sap_lc.flux - median_sap_flux * mean_model(self._coeffs)
        return LightCurve(time=sap_lc.time, flux=flux_hat.reshape(-1))

    def get_cbvs_list(self, method='bayes-factor'):
        """Returns the subsequence of subsequent CBVs that maximizes
        Bayes' factor [1]_.

        Returns
        -------
        cbv_list : list
            Subsequence of subsequent CBVs that maximizes the Bayes' factor.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Bayes_factor
        """

        self.bayes_factor, cost = [], [] # bayes_factor here is actually the
                                         # negative log of the bayes factor
        self.correct(cbvs=[1], options={'xtol': 1e-6, 'ftol':1e-6, 'maxfev': 2000})
        cost.append(self.opt_result.fun)
        for n in tqdm(range(2, self._ncbvs+1)):
            cbv_list = list(range(1, n+1))
            self.correct(cbv_list, options={'xtol': 1e-6, 'ftol':1e-6, 'maxfev': 2000})
            cost.append(self.opt_result.fun)
            # cost is the negative log of the posterior evaluated at the
            # Maximum A Posterior Probability (MAP) estimator
            self.bayes_factor.append((cost[n-2] - cost[n-1]))
            # so cost[n-2] - cost[n-1] = -log(p1) + log(p2) = log(p2/p1)
            # where p1 is the posterior probability (evaluated at the MAP)
            # for the model with n-2 cbvs and p2 is the posterior probability
            # also evaluated at the MAP for the model with n-1 cbvs
        k = np.argmin(self.bayes_factor)
        # transform to get the actual Bayes factor
        self.bayes_factor = np.exp(-np.array(self.bayes_factor))
        # the k+2 here comes from the fact that Python indexes begin
        # from 0 and we count CBVs starting from 1 and also
        # note that range(1, k) equals the interval [1, k), which excludes k.
        return list(range(1, k+2))

    def get_cbv_url(self):
        # gets the html page and finds all references to 'a' tag
        # keeps the ones for which 'href' ends with 'fits'
        # this might slow things down in case the user wants to fit 1e3 stars
        soup = BeautifulSoup(requests.get(self.cbv_base_url).text, 'html.parser')
        cbv_files = [fn['href'] for fn in soup.find_all('a') if fn['href'].endswith('fits')]

        if self.lc_file.mission == 'Kepler':
            if self.lc_file.quarter < 10:
                quarter = 'q0' + str(self.lc_file.quarter)
            else:
                quarter = 'q' + str(self.lc_file.quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif self.lc_file.mission == 'K2':
            if self.lc_file.campaign <= 8:
                campaign = 'c0' + str(self.lc_file.campaign)
            else:
                campaign = 'c' + str(self.lc_file.campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break

        return self.cbv_base_url + cbv_file


class SPLDCorrector(object):
    r"""
    Implements the simple first order Pixel Level Decorrelation (PLD) proposed by
    Deming et. al. [1]_ and Luger et. al. [2]_, [3]_.

    Attributes
    ----------


    Notes
    -----
    This code serves only as a quick look into the PLD technique.
    Users are encouraged to check out the GitHub repos
    `everest <http://www.github.com/rodluger/everest>`_
    and `everest3 <http://www.github.com/rodluger/everest3>`_.

    References
    ----------
    .. [1] Deming et. al. Spitzer Secondary Eclipses of the Dense, \
           Modestly-irradiated, Giant Exoplanet HAT-P-20b using Pixel-Level Decorrelation.
    .. [2] Luger et. al. EVEREST: Pixel Level Decorrelation of K2 Light Curves.
    .. [3] Luger et. al. An Update to the EVEREST K2 Pipeline: short cadence, \
           saturated stars, and Kepler-like photometry down to K_p = 15.
    """

    def __init__(self):
        pass

    def correct(self, time, tpf_flux, window_length=None, polyorder=2):
        """
        Parameters
        ----------
        time : array-like
            Time array
        tpf_flux : array-like
            Pixel values series
        window_length : int
        polyorder : int
        """
        k = window_length
        if not k:
            k = int(len(time) / 2) - 1
        n_windows = int(len(time) / k)
        flux_hat = np.array([])
        for n in range(1, n_windows + 1):
            flux_hat = np.append(flux_hat,
                                 self._pld(tpf_flux[(n - 1) * k:n * k], polyorder))
        flux_hat = np.append(flux_hat, self._pld(tpf_flux[n * k:], polyorder))
        return LightCurve(time, flux_hat + np.nanmedian(np.nansum(tpf_flux, axis=(1, 2))))

    def _pld(self, tpf_flux, polyorder=2):
        if len(tpf_flux) == 0:
            return np.array([])
        pixels_series = tpf_flux.reshape((tpf_flux.shape[0], -1))
        lightcurve = np.nansum(pixels_series, axis=1).reshape(-1, 1)
        # design matrix
        X = pixels_series / lightcurve
        X = np.hstack((X, np.array([np.linspace(0, 1, tpf_flux.shape[0]) ** n for n in range(polyorder+1)]).T))
        opt_weights = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, lightcurve))
        model = np.dot(X, opt_weights)
        flux_hat = lightcurve - model
        return flux_hat


def box_period_search(lc, min_period=0.5, max_period=30, nperiods=2000,
                      prior=None):
    """
    Implements a brute force search to find transit-like periodic events.
    This function fits a "box" model defined as:

    .. math::

        \Pi (t) =
            \left\{
                \begin{array}{ll}
                    a, & t < t_o,\\
                    a - d, & t_o \leq t < t_o + w, \\
                    a, & t \geq t_o + w
                \end{array}
            \right.

    to a list of `nperiods` periods between `min_period` and `max_period`.
    It's assumed that the best period is the one that maximizes the posterior
    probability of the fit.

    Parameters
    ----------
    lc : LightCurve object
        An object from KeplerLightCurve or LightCurve.
        Note that flattening the lightcurve beforehand does aid the quest
        for the transit period.
    min_period : float
        Minimum period to search for. Units must be the same as `lc.time`.
    max_period : float
        Maximum period to search for. Units must be the same as `lc.time`.
    nperiods : int
        Number of periods to search between `min_period` and `max_period`.
    prior : oktopus.Prior object
        Prior probability on the parameters of the box function,
        namely, `amplitude`, `depth`, `to` (time of the first discontinuity),
        and `width`.

    Returns
    -------
    log_posterior : list
        Log posterior (up to an additive constant) of the fit. The "best"
        period is therefore the one that maximizes the log posterior
        probability.
    trial_periods : numpy array
        List of trial periods.
    best_period : float
        Best period.
    """

    def box(amplitude, depth, to, width):
        """A simple box function defined in the interval [-.5, .5].
        `to` is the time of the first discontinuity.
        """
        t = np.linspace(-.5, .5, len(lc.time))
        val = np.zeros(len(lc.time))
        val[t < to] = amplitude
        val[(t >= to) * (t < to + width)] = amplitude - depth
        val[t >= to + width] = amplitude
        return val

    if prior is None:
        prior = oktopus.UniformPrior(lb=[0.9, 0., -.4, 0.],
                                     ub=[1.15, .5, .5, .3])
    lc = lc.normalize()
    log_posterior = []
    trial_periods = np.linspace(min_period, max_period, nperiods)
    for p in tqdm(trial_periods):
        folded = lc.fold(period=p)
        # var should be set to the uncertainty in the data point
        ll = oktopus.GaussianPosterior(data=folded.flux, mean=box, var=1.,
                                       prior=prior)
        res = ll.fit(x0=prior.mean, method='powell',
                     options={'ftol':1e-9, 'xtol':1e-9, 'maxfev': 2000})
        # fun is the negative log posterior
        log_posterior.append(-res.fun)

    return log_posterior, trial_periods, trial_periods[np.argmax(log_posterior)]
