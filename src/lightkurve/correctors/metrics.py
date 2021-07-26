"""Metrics to assess under- and over-fitting of systematic noise.

This module provides two metrics, `overfit_metric_lombscargle` and `underfit_metric_neighbors`,
which enable users to assess whether the noise in a systematics-corrected light curve has been
under- or over-fitted.  These features were contributed by Jeff Smith (cf. https://github.com/lightkurve/lightkurve/pull/855)
and are in turn inspired by similar metrics in use by the PDC module of the official Kepler/TESS pipeline.
"""
import logging
import copy

import numpy as np
from scipy.interpolate import PchipInterpolator
from memoization import cached
from astropy import units as u

from .. import LightCurve


log = logging.getLogger(__name__)


def overfit_metric_lombscargle(
    original_lc: LightCurve, corrected_lc: LightCurve, n_samples: int = 10
) -> float:
    """Uses a LombScarglePeriodogram to assess the change in broad-band
    power in a corrected light curve to measure the degree of over-fitting.

    The to_periodogram Lomb-Scargle method is used and the sampling band is
    from one frequency separation to the Nyquist frequency

    This over-fitting goodness metric is calibrated such that a metric
    value of 0.5 means the introduced noise due to over-fitting is at the
    same power level as the uncertainties in the light curve.

    Parameters
    ----------
    original_lc : LightCurve
        Uncorrected light curve.
    corrected_lc : LightCurve
        Light curve from which systematics have been removed.
    n_samples : int
        The number of times to compute and average the metric
        This can stabilize the value, default = 10

    Returns
    -------
    overfit_metric : float
        A float in the range [0,1] where 0 => Bad, 1 => Good
    """
    # The fit can sometimes result in NaNs
    # Also median normalize original and correctod LCs
    orig_lc = original_lc.copy()
    orig_lc = orig_lc.remove_nans().normalize()
    orig_lc -= 1.0
    corrected_lc = corrected_lc.copy()
    corrected_lc = corrected_lc.remove_nans().normalize()
    corrected_lc -= 1.0
    if len(corrected_lc) == 0:
        return 1.0

    # Perform the measurement multiple times and average to stabilize the metric
    metric_per_iter = []
    for idx in np.arange(n_samples):
        pgOrig = orig_lc.to_periodogram()
        # Use the same periods in the corrected flux as just used in the
        # original flux
        pgCorrected = corrected_lc.to_periodogram(frequency=pgOrig.frequency)

        # Get an estimate of the PSD at the uncertainties limit
        # The raw and corrected uncertainties should be essentially identical so
        # use the corrected
        # TODO: the periodogram of WGN should be analytical to compute!
        nNonGappedCadences = len(orig_lc)
        meanCorrectedUncertainties = np.nanmean(corrected_lc.flux_err)
        WGNCorrectedUncert = (
            np.random.randn(nNonGappedCadences, 1) * meanCorrectedUncertainties
        ).T[0]
        model_err = np.zeros(nNonGappedCadences)
        noise_lc = LightCurve(
            time=orig_lc.time, flux=WGNCorrectedUncert, flux_err=model_err
        )
        pgCorrectedUncert = noise_lc.to_periodogram()
        meanCorrectedUncertPower = np.nanmean(np.array(pgCorrectedUncert.power))

        # Compute the change in power
        pgChange = np.array(pgCorrected.power) - np.array(pgOrig.power)

        # Ignore nans
        pgChange = pgChange[~np.isnan(pgChange)]

        # If no increase in power in ANY bands then return a perfect loss
        # function
        if len(np.nonzero(pgChange > 0.0)[0]) == 0:
            metric_per_iter.append(0.0)
        else:
            # We are only concerned with bands where the power increased so
            # when(pgCorrected - pgOrig) > 0
            # Normalize by the noise in the uncertainty
            # We want the goodness to begin to degrade when the introduced
            # noise is greater than the uncertainties.
            # So, when Sigmoid > 0.5 (given twiceSigmoidInv defn.)
            denominator = (
                len(np.nonzero(pgChange > 0.0)[0])
            ) * meanCorrectedUncertPower
            if denominator == 0:
                # Suppress divide by zero warning
                result = np.inf
            else:
                result = np.sum(pgChange[pgChange > 0.0]) / denominator
            metric_per_iter.append(result)

    metric = np.mean(metric_per_iter)

    # We want the goodness to span (0,1]
    # Use twice a reversed sigmoid to get a [0,1] range mapped from a [0,inf) range
    def sigmoidInv(x):
        return 2.0 / (1 + np.exp(x))

    # Make sure maximum score is 1.0
    metric = sigmoidInv(np.max([metric, 0.0]))

    return metric


def underfit_metric_neighbors(
    corrected_lc: LightCurve,
    radius: float = 6000,
    min_targets: int = 30,
    max_targets: int = 50,
    interpolate: bool = False,
    extrapolate: bool = False,
):
    """This goodness metric measures the degree of under-fitting of the
    CBVs to the light curve. It does so by measuring the mean residual target to
    target Pearson correlation between the target under study and a selection of
    neighboring SPOC SAP target light curves.

    This function will search within the given radiu in arceseconds and find the
    min_targets nearest targets up until max_targets is reached. If less than
    min_targets is found a MinTargetsError Exception is raised.

    The downloaded neighboring targets will normally be "aligned" to the
    corrected_lc, meaning the cadence numbers are used to align the targets
    to the corrected_lc. However, if interpolate=True then the targets will be
    interpolated to the corrected_lc cadence times. extrapolate=True will 
    further extrapolate the targets to the corrected_lc cadence times.

    The returned under-fitting goodness metric is callibrated such that a
    value of 0.95 means the residual correlations in the target is
    equivalent to chance correlations of White Gaussian Noise.

    Parameters
    ----------
    corrected_lc : LightCurve
        Light curve from which systematics have been removed.
    radius : float
        Search radius to find neighboring targets in arcseconds
    min_targets : int
        Minimum number of targets to use in correlation metric
        Using too few can cause unreliable results. Default = 30
    max_targets : int
        Maximum number of targets to use in correlation metric
        Using too many can slow down the metric due to large data
        download. Default = 50
    interpolate : bool
        If `True`, the flux values of the neighboring light curves will be
        interpolated to match the times of the `corrected_lc`.
        If `False`, the flux values will simply be aligned by time where possible.

    Returns
    -------
    under_fitting_metric : float
        A float in the range [0,1] where 0 => Bad, 1 => Good
    """

    # Normalize and condition the corrected light curve
    corrected_lc = corrected_lc.copy().remove_nans().normalize()
    corrected_lc -= 1.0
    corrected_lc_flux = corrected_lc.flux.value

    # Download and pre-process neighboring light curves
    lc_neighborhood, lc_neighborhood_flux = _download_and_preprocess_neighbors(
        corrected_lc=corrected_lc,
        radius=radius,
        min_targets=min_targets,
        max_targets=max_targets,
        interpolate=interpolate,
        extrapolate=extrapolate,
        flux_column="sap_flux",
    )

    # Create fluxMatrix. The last entry is the target under study
    # Check that all neighboring targets have similar shape
    if not np.all([len(lc_neighborhood_flux[0]) == len(l) for l in lc_neighborhood_flux]):
        raise Exception('Neighbroing targets do not all have the same shape')
    fluxMatrix = np.zeros((len(lc_neighborhood_flux[0]), len(lc_neighborhood_flux) + 1))
    for idx in np.arange(len(fluxMatrix[0, :]) - 1):
        fluxMatrix[:, idx] = lc_neighborhood_flux[idx]
    # Add in the trimmed target under study
    fluxMatrix[:, -1] = corrected_lc_flux

    # Ignore NaNs
    mask = ~np.isnan(corrected_lc_flux)
    fluxMatrix = fluxMatrix[mask, :]

    # Determine the target-target correlation between target and
    # neighborhood
    correlationMatrix = _compute_correlation(fluxMatrix)

    # The selection basis for targets used for the PDC-MAP SVD  uses median
    # absolute correlation per star.  However, here we wish to overemphasize
    # any residual correlation between a handfull of targets and not the
    # overall correlation (which should almost always be low).

    # We want a residual correlation larger than random correlations of WGN
    # to mean a meaningful correlation. The median Pearson correlation of
    # WGN of nCadences is approximated by the equation:
    # 0.0010288 + 0.80304 nCadences^ -0.50128
    nCadences = len(fluxMatrix[:, 0])
    beta = [0.0007, 0.8083, -0.5023]
    WGNCorrelation = beta[0] + beta[1] * (nCadences ** (beta[2]))

    # badLimit is the goodness value for WGN correlations
    # I.e. anything above this goodness value is equivalent to random correlations
    # I.e. 0.95 = sigmoidInv(WGNCorr * correlationScale)
    badLimit = 0.95
    correlationScale = 1 / (WGNCorrelation) * np.log((2.0 / badLimit) - 1.0)

    # Over-emphasize any individual correlation groups. Note the power of
    # three after taking the absolute value
    # of the correlation. Also, the mean is used so that outliers are *not* ignored.
    # Zero diagonal elements
    correlationMatrix = np.tril(correlationMatrix, k=-1) + np.triu(
        correlationMatrix, k=+1
    )

    # Add up the correlation over all targets ignoring NaNs (no corrected fit)
    correlation = correlationScale * np.nanmean(np.abs(correlationMatrix) ** 3, axis=0)

    # We only want the last entry, which is for the target under study
    correlation = correlation[-1]

    # We want the goodness to span (0,1]
    # Use twice a reversed sigmoid to get a [0,1] range mapped from a [0,inf) range
    def sigmoidInv(x):
        return 2.0 / (1 + np.exp(x))

    metric = sigmoidInv(correlation)

    return metric


# Custom exception to track when minimum targets is not reached
class MinTargetsError(Exception):
    pass


def _unique_key_for_processing_neighbors(
    corrected_lc: LightCurve,
    radius: float = 6000.0,
    min_targets: int = 30,
    max_targets: int = 50,
    interpolate: bool = False,
    extrapolate: bool = False,
    author: tuple = ("Kepler", "K2", "SPOC"),
    flux_column: str = "sap_flux",
):
    """Returns a unique key that will determine whether a cached version of a
    call to `_download_and_preprocess_neighbors` can be re-used."""
    return f"{corrected_lc.ra}{corrected_lc.dec}{corrected_lc.cadenceno}{radius}{min_targets}{max_targets}{author}{flux_column}{interpolate}"


@cached(custom_key_maker=_unique_key_for_processing_neighbors)
def _download_and_preprocess_neighbors(
    corrected_lc: LightCurve,
    radius: float = 6000.0,
    min_targets: int = 30,
    max_targets: int = 50,
    interpolate: bool = False,
    extrapolate: bool = False,
    author: tuple = ("Kepler", "K2", "SPOC"),
    flux_column: str = "sap_flux",
):
    """Returns a collection of neighboring light curves.

    If less than min_targets a MinTargetsError Exception is raised.

    Parameters
    ----------
    corrected_lc : LightCurve
        Light curve around which to look for neighbors.
    radius : float
        Conesearch radius in arcseconds.
    min_targets : int
        Minimum number of targets to return.
        A `ValueError` will be raised if this number cannot be obtained.
    max_targets : int
        Maximum number of targets to return.
        Using too many can slow down this function due to large data
        download.
    interpolate : bool
        If `True`, the flux values of the neighboring light curves will be
        interpolated to match the times of the `corrected_lc`.
        If `False`, the flux values will simply be aligned by time where possible.
    extrapolate : bool
        If `True`, the  flux values of the neighboring light curves will be
        also be extrapolated. Note: extrapolated values can be unstable.

    Returns
    -------
    lc_neighborhood : LightCurveCollection
        Collection of all neighboring light curves used.
    lc_neighborhood_flux : list
        List containing the flux arrays of the neighboring light curves,
        interpolated or aligned with `corrected_lc` if requested.
    """
    if extrapolate and (extrapolate != interpolate):
        raise Exception('interpolate must be True if extrapolate is True')

    search = corrected_lc.search_neighbors(
        limit=max_targets, radius=radius, author=author
    )
    if len(search) < min_targets:
        raise MinTargetsError(
            f"Unable to find at least {min_targets} neighbors within {radius} arcseconds radius."
        )
    log.info(
        f"Downloading {len(search)} neighboring light curves. This might take a while."
    )
    lcfCol = search.download_all(flux_column=flux_column)

    # Pre-process the neighboring light curves
    # Align or interpolate to the corrected light curve
    lc_neighborhood = []
    lc_neighborhood_flux = []
    # Extract SAP light curves
    # We want zero-centered median normalized light curves
    for lc in lcfCol:
        lcSAP = lc.remove_nans().normalize()
        lcSAP.flux -= 1.0
        # Align or interpolate the neighboring target with the target under study
        if interpolate:
            # Interpolate to corrected_lc cadence times
            fInterp = PchipInterpolator(
                lcSAP.time.value,
                lcSAP.flux.value,
                extrapolate=extrapolate,
            )
            lc_neighborhood_flux.append(fInterp(corrected_lc.time.value))
        else:
            # The CBVs were aligned so also align the neighboring
            # lightcurves
            aligned_lcSAP = _align_to_lc(lcSAP, corrected_lc)
            lc_neighborhood_flux.append(aligned_lcSAP.flux.value)

        lc_neighborhood.append(lcSAP)

    if len(lc_neighborhood) < min_targets:
        raise MinTargetsError(
            f"Unable to find at least {min_targets} neighbors within {radius} arcseconds radius."
        )
    # Store the unmolested lightcurve neighborhood but also save the
    # aligned or interpolated neighborhood flux
    from .. import LightCurveCollection  # local import to avoid circular import

    lc_neighborhood = LightCurveCollection(lc_neighborhood)
    lc_neighborhood_flux = lc_neighborhood_flux

    return lc_neighborhood, lc_neighborhood_flux

def _align_to_lc(lc, ref_lc):
    """ Aligns a light curve to a reference light curve.

    This method will use the cadence number (lc.cadenceno) to
    perform the synchronization. Only cadence numbers that exist in both
    the lc and the ref_lc will have values in the returned lc. All
    cadence numbers that exist in ref_lc but not in lc will
    have NaNs returned for those cadences.

    Any cadences in the lc not in ref_lc will be removed from the returnd lc.

    The returned lc is sorted by cadenceno.

    Parameters
    ----------
    lc : LightCurve object
        The light curve to align
    ref_lc : LightCurve object
        The reference light curve to align to

    Returns
    -------
    lc : LightCurve object
        The light curve aligned to ref_lc
    """

    if not isinstance(lc, LightCurve):
        raise Exception('<lc> must be a LightCurve class')
    if not isinstance(ref_lc, LightCurve):
        raise Exception('<ref_lc> must be a LightCurve class')

    if hasattr(lc, 'cadenceno'):

        # Make a deepcopy so we do not just return a modified original
        aligned_lc = copy.deepcopy(lc)

        # NaN any cadences in ref_lc and not lc
        # This requires us to add rows to the lc table
        lc_nan_mask = np.logical_not(np.in1d(ref_lc.cadenceno, aligned_lc.cadenceno))
        lc_nan_indices = np.nonzero(lc_nan_mask)[0]
        if len(lc_nan_indices) > 0:
            row_to_add = LightCurve(aligned_lc[0:len(lc_nan_indices)])
            row_to_add['time'] = ref_lc.time[lc_nan_indices]
            row_to_add['cadenceno'] = ref_lc.cadenceno[lc_nan_indices]
            row_to_add['flux'] = np.nan
            aligned_lc = aligned_lc.append(row_to_add)

        # There appears to be a bug in astropy.timeseries when using ts[x:y]
        # in combination with ts.remove_row() or ts.remove_rows.
        # See LightKurve Issue #836.
        # To get around the error for now, we will attempt to use
        # ts[x:y]. If it errors out then revert to remove_rows, which is
        # REALLY slow.
        try:
            # This method is fast but might cause errors
            keep_indices = np.nonzero(np.in1d(aligned_lc.cadenceno, ref_lc.cadenceno))[0]
            aligned_lc = aligned_lc[keep_indices]
        except:
            # This method is slow but appears to be more robust
            trim_indices = np.nonzero(np.logical_not(
                np.in1d(aligned_lc.cadenceno, ref_lc.cadenceno)))[0]
            aligned_lc.remove_rows(trim_indices)

        # Now sort the lc by cadenceno
        aligned_lc.sort('cadenceno')

    else:
        raise Exception('align requires cadence numbers for the ' + \
                'light curve. NO ALIGNMENT OCCURED')

    return aligned_lc


def _compute_correlation(fluxMatrix):
    """Finds the empirical target to target flux time series Pearson correlation.

    Parameters
    ----------
    fluxMatrix : float 2-d array[ntargets,ncadences]
        The matrix of target flux. There should be no gaps or Nans

    Returns
    -------
    correlation_matrix : [float 2-d array] (nTargets x nTargets)
        The target-target correlation
    """

    nCadences = len(fluxMatrix[:, 0])

    # Scale each flux value by the RMS flux for the given target.
    rmsFlux = np.sqrt(np.sum(fluxMatrix ** 2.0, axis=0) / nCadences)
    # If RMS is zero then set to Inf so that we don't get a divide by zero warning
    rmsFlux[np.nonzero(rmsFlux == 0.0)[0]] = np.inf
    unitNormFlux = fluxMatrix / np.tile(rmsFlux, (nCadences, 1))

    correlation_matrix = unitNormFlux.T.dot(unitNormFlux) / nCadences

    return correlation_matrix
