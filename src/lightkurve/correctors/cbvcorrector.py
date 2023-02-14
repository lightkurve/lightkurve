"""Defines Corrector classes that utilize Kepler/K2/TESS Cotrending Basis Vectors.
"""
import logging
import copy
import requests
import urllib.request
import glob
import os
import warnings

from astropy.io import fits as pyfits
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.units import Quantity, Unit, UnitsWarning
from astropy.utils.decorators import deprecated
from astropy.utils.masked import Masked

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn import linear_model
from scipy.optimize import minimize_scalar

from .designmatrix import DesignMatrix, DesignMatrixCollection
from .. import MPLSTYLE
from ..lightcurve import LightCurve
from ..utils import channel_to_module_output, validate_method, LightkurveDeprecationWarning
from ..search import search_lightcurve
from .regressioncorrector import RegressionCorrector
from ..collections import LightCurveCollection
from .metrics import overfit_metric_lombscargle, underfit_metric_neighbors, MinTargetsError


log = logging.getLogger(__name__)

__all__ = ['CBVCorrector', 'CotrendingBasisVectors', 'KeplerCotrendingBasisVectors',
           'TessCotrendingBasisVectors', 'load_kepler_cbvs','load_tess_cbvs',
           'download_kepler_cbvs', 'download_tess_cbvs']

#*******************************************************************************
# CBV Corrector Class

class CBVCorrector(RegressionCorrector):
    """Class for removing systematics using Cotrending Basis Vectors (CBVs)
    from Kepler/K2/TESS.

    On construction of this object, the relevant CBVs will be downloaded from
    MAST appropriate for the lightcurve object passed to the constructor.

    For TESS there are multiple CBV types. All are loaded and the user must
    specify which to use in the correction.

    Attributes
    ----------
    lc  : LightCurve
        The light curve loaded into CBVCorrector in electrons / second
    cbvs : CotrendingBasisVectors list
        The retrieved CBVs, can contain multiple types of CBVs
    interpolated_cbvs : bool
        If true then the CBVs have been interpolated to the lightcurve
    cbv_design_matrix : DesignMatrix
        The retrieved CBVs ported into a DesignMatrix object
    extra_design_matrix : DesignMatrix
        An extra design matrix to include in the fit with the CBVs
    design_matrix_collection : DesignMatrixCollection
        The design matrix collection composed of cbv_design_matrix and extra_design_matrix
    corrected_lc : LightCurve
        The returned light curve from correct() in electrons / second
    coefficients : float ndarray
        The fit coefficients corresponding to the design_matrix_collection
    coefficients_err : float ndarray
        The error estimates for the coefficients, see regressioncorrector
    model_lc : LightCurve
        The model fit to the lightcurve 'lc'
    diagnostic_lightcurves : dict
        Model fits for each of the sub design matrices fit in model_lc
    lc_neighborhood : LightCurveCollection
        SPOC SAP light curves of all targets within the defined neighborhood of the
        target under study for use with the under-fitting metric
    lc_neighborhood_flux : list of arrays
        Neighboring target flux aligned or interpolated to the target under
        study cadence
    cadence_mask : np.ndarray of bool
        Mask, where True indicates a cadence that was used in
        RegressionCorrector.correct.
        Note: The saved cadence_mask is overwritten for each call to correct().
    over_fitting_score : float
        Over-fitting score from the most recent run of correct()
    under_fitting_score : float
        Under-fitting score from the most recent run of correct()
    alpha : float
        L2-norm regularization term used in most recent fit
        Equivalent to: designmatrix prior sigma = np.median(self.lc.flux_err) / np.sqrt(alpha)
    """

    def __init__(self, lc, interpolate_cbvs=False, extrapolate_cbvs=False, do_not_load_cbvs=False,
        cbv_dir=None):
        """Constructor

        This constructor will retrieve all relevant CBVs from MAST and then
        align or interpolate them with the passed-in light curve.

        Parameters
        ----------
        lc  : LightCurve
            The light curve to correct
        interpolate_cbvs : bool
            By default, the cbvs will be 'aligned' to the lightcurve. If you
            wish to interpolate the cbvs instead then set this to True.
            Uses Piecewise Cubic Hermite Interpolating Polynomial (PCHIP). 
        extrapolate_cbvs : bool
            Set to True if the CBVs also have to be extrapolated outside their time 
            stamp range. (If False then those cadences are filled with NaNs.)
        do_not_load_cbvs : bool
            If True then the CBVs will NOT be loaded from MAST. 
            Use this option if you wish to use the CBV corrector methods with only a 
            custom design matrix (via the ext_dm argument in the corrector methods)
        cbv_dir : str
            Path to specific directory holding TESS CBVs. If this is None, will query
            MAST by default.
        """
        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        assert  lc.flux.unit==Unit('electron / second'), \
            'cbvCorrector expects light curve to be passed in e-/s units.'        

        if extrapolate_cbvs and (extrapolate_cbvs != interpolate_cbvs):
            raise Exception('interpolate_cbvs must be True if extrapolate_cbvs is True')

        # We do not want any NaNs
        lc = lc.remove_nans()

        # Call the RegresssionCorrector Constructor
        super(CBVCorrector, self).__init__(lc)

        #***
        # Retrieve all relevant CBVs from either MAST or a local directory
        cbvs = []

        if (not do_not_load_cbvs):
            if self.lc.mission == 'Kepler':
                cbvs.append(load_kepler_cbvs(cbv_dir=cbv_dir,mission=self.lc.mission, quarter=self.lc.quarter,
                        channel=self.lc.channel))
            elif self.lc.mission == 'K2':
                cbvs.append(load_kepler_cbvs(cbv_dir=cbv_dir,mission=self.lc.mission, campaign=self.lc.campaign,
                        channel=self.lc.channel))
            elif self.lc.mission == 'TESS':
                # For TESS we load multiple CBV types
                # Single-Scale
                cbvs.append(load_tess_cbvs(cbv_dir=cbv_dir,sector=self.lc.sector,
                    camera=self.lc.camera, ccd=self.lc.ccd, cbv_type='SingleScale'))
            
                # Multi-Scale
                # Although there has always been 3 bands, there could be more,
                # continue to load more bands until no more are left to load
                iBand = int(0)
                moreData = True
                while moreData:
                    iBand += 1
                    cbvObj = load_tess_cbvs(cbv_dir=cbv_dir,sector=self.lc.sector,
                        camera=self.lc.camera, ccd=self.lc.ccd, cbv_type='MultiScale',
                        band=iBand)
                    if (cbvObj.band == iBand):
                        cbvs.append(cbvObj)
                    else:
                        moreData = False
            
                # Spike
                cbvs.append(load_tess_cbvs(cbv_dir=cbv_dir,sector=self.lc.sector,
                    camera=self.lc.camera, ccd=self.lc.ccd, cbv_type='Spike'))
            
            else:
                raise ValueError('Unknown mission type')
            
            for idx in np.arange(len(cbvs)):
                if (not isinstance(cbvs[idx], CotrendingBasisVectors)):
                    raise Exception('CBVs could not be loaded. CBVCorrector must exit')

            # Set the CBV time format units to the lightcurve time format units
            for idx in np.arange(len(cbvs)):
                # astropy.time.Time makes this easy!
                cbvs[idx].time.format = lc.time.format
            
            # Align or interpolate the CBVs with the lightcurve flux using the cadence numbers
            for idx in np.arange(len(cbvs)):
                if interpolate_cbvs:
                    cbvs[idx] = cbvs[idx].interpolate(self.lc, extrapolate=extrapolate_cbvs)
                else:
                    cbvs[idx] = cbvs[idx].align(self.lc)

        self.cbvs = cbvs
        self.interpolated_cbvs = interpolate_cbvs
        self.extrapolated_cbvs = extrapolate_cbvs

        # Initialize all extra attributes to None
        self.cbv_design_matrix = None
        self.extra_design_matrix = None
        self.design_matrix_collection = None
        self.corrected_lc = None
        self.coefficients = None
        self.coefficients_err = None
        self.model_lc = None
        self.diagnostic_lightcurves = None
        self.lc_neighborhood = None
        self.lc_neighborhood_flux = None
        self.cadence_mask = None
        self.over_fitting_score = None
        self.under_fitting_score = None
        self.alpha = None

    def correct_gaussian_prior(self, cbv_type=['SingleScale'],
            cbv_indices=[np.arange(1,9)], 
            alpha=1e-20, ext_dm=None, cadence_mask=None, **kwargs):
        """ Performs the correction using RegressionCorrector methods.

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix (ext_dm). It then uses the
        alpha L2-Norm (Ridge Regression) penalty term to set the width on the
        design matrix priors.  Then uses the super-class
        RegressionCorrector.correct to perform the correction.

        The relation between the L2-Norm alpha term and the Gaussian prior sigma
        is: 
        alpha = flux_sigma^2 / sigma^2

        By default this method will use the first 8 "SingleScale" basis vectors.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha : float
            L2-norm regularization penatly term. Default = 1e-20
            {0 => no regularization}
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        **kwargs : dict
            Additional keyword arguments passed to
            `RegressionCorrector.correct`.

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the correction using the
        SingleScale and Spike basis vectors with a weak regularization alpha
        term of 0.1. It also adds in an external design matrix to perfomr a
        joint fit.
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> corrected_lc = cbvCorrector.correct_gaussian_prior(cbv_type=cbv_type, # doctest: +SKIP
            >>>     cbv_indices=cbv_indices, alpha=0.1, # doctest: +SKIP
            >>>     ext_dm=design_matrix ) # doctest: +SKIP
        """
        
        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Add in a width to the Gaussian priors
        # alpha = flux_sigma^2 / sigma^2
        if (alpha == 0.0):
            sigma = None
        else:
            sigma = np.median(self.lc.flux_err.value) / np.sqrt(np.abs(alpha))
        self._set_prior_width(sigma)
            
        # Use RegressionCorrector.correct for the actual fitting
        self.correct_regressioncorrector(self.design_matrix_collection, 
                cadence_mask=cadence_mask, **kwargs)

        self.alpha = alpha

        return self.corrected_lc

    def correct_elasticnet(self, cbv_type='SingleScale', cbv_indices=np.arange(1,9), 
            alpha=1e-20, l1_ratio=0.01, ext_dm=None, cadence_mask=None, **kwargs):
        """ Performs the correction using scikit-learn's ElasticNet which
        utilizes combined L1- and L2-Norm priors as a regularizer.

        This method will assemble the full design matrix collection composed of
        cbv_design_matrix and extra_design_matrix (ext_dm). Then uses
        scikit-learn.linear_model.ElasticNet to perform the correction.

        By default this method will use the first 8 "SingleScale" basis vectors.

        This method will preserve the median value of the light curve flux.

        Note that the alpha term in scikit-learn's ElasticNet does not have the
        same scaling as when used in CBVCorrector.correct_gaussian_prior or 
        CBVCorrector.correct. Do not assume similar results with a
        similar alpha value.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha : float
            L2-norm regularization pentaly term.
            {0 => no regularization}
        l1_ratio : float
            Elastic-Net mixing parameter
            l1_ratio = 0 => L2 penalty (Ridge). l1_ratio = 1 => L1 penalty (Lasso).
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        **kwargs : dict
            Additional keyword arguments passed to
            `sklearn.linear_model.ElasticNet`.

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the ElasticNet correction using the
        SingleScale and Spike basis vectors with a strong regualrization alpha
        term of 1.0 and an L1 ratio of 0.9 which means predominantly a Lasso
        regularization but with a slight amount of Ridge Regression.
            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> corrected_lc = cbvCorrector.correct_elasticnet(cbv_type=cbv_type, # doctest: +SKIP
            >>>     cbv_indices=cbv_indices, alpha=1.0, l1_ratio=0.9) # doctest: +SKIP
        """
        
        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Default cadence mask
        if cadence_mask is None:
            cadence_mask = np.ones(len(self.lc.flux), bool)

        # Use Scikit-learn ElasticNet
        self.regressor = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                fit_intercept=False, **kwargs)

        X = self.design_matrix_collection.values
        y = self.lc.flux

        # Set mask
        # note: ElasticNet has no internal way to do this so we have to just
        # remove the cadences from X and y
        XMasked = X.copy()
        yMasked = y.copy()
        XMasked = XMasked[cadence_mask,:]
        yMasked = yMasked[cadence_mask]

        # Perform the ElasticNet fit
        self.regressor.fit(XMasked, yMasked)

        # Finishing work
        # When creating the model do not include the constant
        model_flux  = np.dot(X[:,0:-1], self.regressor.coef_[0:-1])
        model_flux -= np.median(model_flux)
        # TODO: Propagation of uncertainties. They really do not change much.
        model_err   = np.zeros(len(model_flux))
        
        self.coefficients = self.regressor.coef_
        
        self.model_lc = LightCurve(time=self.lc.time,
                flux=model_flux*self.lc.flux.unit,
                flux_err=model_err*self.lc.flux_err.unit)
        self.corrected_lc = self.lc.copy()
        self.corrected_lc.flux = self.lc.flux - self.model_lc.flux
        self.corrected_lc.flux_err = (self.lc.flux_err**2 + model_err**2)**0.5
        self.diagnostic_lightcurves = self._create_diagnostic_lightcurves()
        self.cadence_mask = cadence_mask
        self.alpha = alpha
            
        return self.corrected_lc

    def correct(self, cbv_type=['SingleScale'],
            cbv_indices=[np.arange(1,9)], 
            ext_dm=None, cadence_mask=None, alpha_bounds=[1e-4,1e4], 
            target_over_score=0.5, target_under_score=0.5, max_iter=100):
        """ Optimizes the correction by adjusting the L2-Norm (Ridge Regression)
        regularization penalty term, alpha, based on the introduced noise
        (over-fitting) and residual correlation (under-fitting) goodness
        metrics. The numercial optimization is performed using the
        scipy.optimize.minimize_scalar Brent's method.

        The optimizer attempts to maximize the over- and under-fitting goodness
        metrics.  However, once the target_over_score or target_under_score is
        reached, a "Leaky ReLU" is used so that the optimization "pressure"
        concentrates on the other metric until both metrics rise above their
        respective target scores, instead of driving a single metric to near
        1.0.

        The optimization parameters used are stored in self.optimization_params
        as a record of how the optimization was performed.

        The optimized correction is performed using LightKurve's 
        RegressionCorrector methods. See correct_gaussian_prior for details.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use in correction {'ALL' => Use all}
        cbv_indices : list of lists
            List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        alpha_bounds : float list(len=2)
            upper anbd lowe bounds for alpha
        target_over_score : float
            Target Over-fitting metric score
        target_under_score : float
            Target under-fitting metric score
        max_iter : int
            Maximum number of iterations to optimize goodness metrics

        Returns
        -------
        `.LightCurve`
            Corrected light curve, with noise removed. In units of electrons / second

        Examples
        --------
        The following example will perform the correction using the
        SingleScale and Spike basis vectors. It will use alpha bounds of
        [1.0,1e3]. The target over-fitting score is 0.5 and the target
        under-fitting score is 0.8.

            >>> cbv_type = ['SingleScale', 'Spike']
            >>> cbv_indices = [np.arange(1,9), 'ALL']
            >>> cbvCorrector.correct(cbv_type=cbv_type, cbv_indices=cbv_indices,  # doctest: +SKIP
            >>>     alpha_bounds=[1.0,1e3],  # doctest: +SKIP
            >>>     target_over_score=0.5, target_under_score=0.8) # doctest: +SKIP
        """

        # Perform all the preparatory stuff common to all correct methods
        self._correct_initialization(cbv_type=cbv_type,
                cbv_indices=cbv_indices, ext_dm=ext_dm)

        # Create a dictionary for optimization parameters to easily pass to the
        # objective function, and also to save for posterity
        self.optimization_params = {'alpha_bounds': alpha_bounds,
                                    'target_over_score': target_over_score,
                                    'target_under_score': target_under_score,
                                    'max_iter': max_iter,
                                    'cadence_mask': cadence_mask,
                                    'over_metric_nSamples': 1}

        #***
        # Use scipy.optimize.minimize_scalar
        # Minimize the introduced metric
        minimize_result = minimize_scalar(self._goodness_metric_obj_fun, method='Bounded',
                bounds=alpha_bounds,
                options={'maxiter':max_iter, 'disp': False})

        # Re-fit with final alpha value
        # (scipy.optimize.minimize_scalar does not exit with the final fit!)
        self._goodness_metric_obj_fun(minimize_result.x)

        # Only display over- or under-fitting scores if requested to optimize
        # for each
        if (self.optimization_params['target_over_score'] > 0):
            self.over_fitting_score = self.over_fitting_metric(n_samples=10)
            print('Optimized Over-fitting metric: {}'.format(self.over_fitting_score))
        else:
            self.over_fitting_score = -1.0

        if (self.optimization_params['target_under_score'] > 0):
            self.under_fitting_score = self.under_fitting_metric()
            print('Optimized Under-fitting metric: {}'.format(self.under_fitting_score))
        else:
            self.under_fitting_score = -1.0

        self.alpha = minimize_result.x

        print('Optimized Alpha: {0:2.3e}'.format(self.alpha))

        return self.corrected_lc

    def correct_regressioncorrector(self, design_matrix_collection, **kwargs):
        """ Pass-through method to gain access to the superclass 
        RegressionCorrector.correct() method.
        """

        # All this does is call the superclass 'correct' method as pass the
        # input arguments.
        return super(CBVCorrector, self).correct(design_matrix_collection, **kwargs)

    def over_fitting_metric(self, 
            n_samples: int = 10):
        """ Computes the over-fitting metric using 
        metrics.overfit_metric_lombscargle

        See that function for a description of the algorithm.

        Parameters
        ----------
        n_samples : int
            The number of times to compute and average the metric
            This can stabalize the value, defaut = 10

        Returns
        -------
        over_fitting_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """

        # Check if corrected_lc is present
        if (self.corrected_lc is None):
            log.warning('A corrected light curve does not exist, please run '
                    'correct first')
            return None

        # Ignore masked cadences
        orig_lc         = self.lc.copy()
        orig_lc         = orig_lc[self.cadence_mask]
        corrected_lc    = self.corrected_lc.copy()
        corrected_lc    = corrected_lc[self.cadence_mask]

        return overfit_metric_lombscargle (orig_lc, corrected_lc, n_samples=n_samples)

    def under_fitting_metric(self, 
            radius: float = None, 
            min_targets: int = 30,
            max_targets: int = 50):
        """  Computes the under-fitting metric using 
        metrics.underfit_metric_neighbors

        See that function for a description of the algorithm.

        For TESS, the default radius is 5000 arcseconds.
        For Kepler/K2, the default radius is 1000 arcseconds

        This function will begin with the given radius in arcseconds and
        finds all neighboring targets. If not enough were found (< min_targets)
        the radius is increased until a minimum number are found.

        The downloaded neighboring targets will be "aligned" to the
        corrected_lc, meaning the cadence numbers are used to align the targets
        to the corrected_lc. However, if the CBVCorrector object was
        instantiated with interpolated_cbvs=True then the targets will be
        interpolated to the corrected_lc cadence times.

        Parameters
        ----------
        radius : float
            Search radius to find neighboring targets in arcseconds
        min_targets : float
            Minimum number of targets to use in correlation metric
            Using too few can cause unreliable results. Default = 30
        max_targets : float
            Maximum number of targets to use in correlation metric
            Using too many can slow down the metric due to large data
            download. Default = 50

        Returns
        -------
        under_fitting_metric : float
            A float in the range [0,1] where 0 => Bad, 1 => Good
        """

        # Check if corrected_lc is present
        if (self.corrected_lc is None):
            raise Exception('A corrected light curve does not exist, please run '
                    'correct first')
            return None

        # Set default radius if one is not provided.
        if (radius is None):
            if (self.lc.mission == 'TESS'):
                radius = 5000
            else:
                radius = 1000

        interpolate = self.interpolated_cbvs
        extrapolate = self.extrapolated_cbvs

        # Make a copy of radius because it changes locally
        dynamic_search_radius = radius
        # Max search radius is the diagonal distance along a CCD in arcseconds
        # 1 pixel in TESS is 21.09 arcseconds
        # 1 pixel in Kepler/K2 is 3.98 arcseconds
        if (self.lc.mission == 'TESS'):
            # 24 degrees of a TESS CCD array (2 CCD's wide) is 86,400 arcseconds
            max_search_radius = np.sqrt(2) * (86400/2.0)
        elif (self.lc.mission == 'Kepler' or self.lc.mission == 'K2'):
            # One Kepler CCD spans 4,096 arcseconds
            max_search_radius = np.sqrt(2) * 4096
        else:
            raise Exception('Unknown mission')

        # Ignore masked cadences
        corrected_lc    = self.corrected_lc.copy()
        corrected_lc    = corrected_lc[self.cadence_mask]
        
        # Dynamically increase radius until min_targets reached.
        continue_searching = True
        while (continue_searching):
            try:
                metric = underfit_metric_neighbors (corrected_lc, 
                            dynamic_search_radius, min_targets, max_targets, 
                            interpolate, extrapolate)
            except MinTargetsError:
                # Too few targets found, try increasing search radius
                if (dynamic_search_radius > max_search_radius):
                    # Hit the edge of the CCD, we have to give up
                    raise Exception('Not enough neighboring targets were '
                        'found. under_fitting_metric failed')
                # Too few found, increase search radius
                dynamic_search_radius *= 1.5
            else:
                continue_searching = False

        return metric

    def _correct_initialization(self, cbv_type='SingleScale', cbv_indices='ALL',
            ext_dm=None):
        """ Performs all the preparatory work needed before applying a 'correct'
        method.

        This helper function is used so that multiple correct methods can be used
        without the need to repeat preparatory code.

        The main thing this method does is set up the design matrix, given the
        requested CBVs and external design matrix.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use
            Can be None if only ext_dm is used
        cbv_indices : list of lists
            List of CBV vectors to use in each passed cbv_type. {'ALL' => Use all}
            Can be None if only ext_dm is used
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to additionally be used in the fit
        """

        assert not ((cbv_type is None) ^ (cbv_indices is None)), \
                'Both cbv_type and cbv_indices must be None, or neither'

        if (cbv_type is None and cbv_indices is None):
            use_cbvs = False
        else:
            use_cbvs = True

        # If any DesignMatrix was passed then store it
        self.extra_design_matrix = ext_dm
        # Check that extra design matrix is aligned with lc flux
        if ext_dm is not None:
            assert isinstance(ext_dm, DesignMatrix), \
                'ext_dm must be a DesignMatrix'
            if (ext_dm.df.shape[0] != len(self.lc.flux)):
                    raise ValueError(
                        'ext_dm must contain the same number of cadences as lc.flux')
            

        # Create a CBV design matrix for each CBV set requested
        self.cbv_design_matrix = []

        if use_cbvs:
            assert (not isinstance(cbv_type, str) and 
                    not isinstance(cbv_indices[0], int)), \
                    'cbv_type and cbv_indices must be lists of strings'
            
            if (self.lc.mission in ['Kepler', 'K2']):
                assert len(cbv_type) == 1 , \
                    'cbv_type must be only Single-Scale for Kepler and K2 missions'
                assert cbv_type == ['SingleScale'], \
                    'cbv_type must be Single-Scale for Kepler and K2 missions'
            
            if (isinstance(cbv_type, list) and len(cbv_type) != 1):
                assert (self.lc.mission == 'TESS'), \
                    'Multiple CBV types are only allowed for TESS'
            
            assert (len(cbv_type) == len(cbv_indices)), \
                'cbv_type and cbv_indices must be the same list length'


            # Loop through all the stored CBVs and find the ones matching the
            # requested cbv_type list
            for idx in np.arange(len(cbv_type)): 
                for cbvs in self.cbvs:
            
                    # Temporarily copy the cbv_indices requested
                    cbv_idx_loop = cbv_indices[idx]
            
                    # If requesting 'ALL' CBVs then set to max default number
                    # Remember, cbv indices is 1-based!
                    if (isinstance(cbv_idx_loop, str) and (cbv_idx_loop == 'ALL')):
                        cbv_idx_loop = cbvs.cbv_indices
                    # Trim to nCBVs in cbvs
                    cbv_idx_loop = np.array([idx for idx in cbv_idx_loop if
                        bool(np.in1d(idx, cbvs.cbv_indices))])
            
                    if cbv_type[idx].find('MultiScale') >= 0:
                        # Find the correct band if this is a multi-scale CBV set
                        band = int(cbv_type[idx][-1])
                        if (cbvs.cbv_type in cbv_type[idx] and cbvs.band == band):
                            self.cbv_design_matrix.append(cbvs.to_designmatrix(
                                cbv_indices=cbv_idx_loop, name=cbv_type[idx]))

                    else:
                        if (cbvs.cbv_type in cbv_type[idx]):
                            self.cbv_design_matrix.append(cbvs.to_designmatrix(
                                cbv_indices=cbv_idx_loop, name=cbv_type[idx]))

        #***
        # Create the design matrix collection with CBVs, plus extra passed basis vectors

        # Create the full design matrix collection from all the sub-design
        # matrices (I.e 'flatten' the design matrix collection)
        if self.extra_design_matrix is not None and \
            self.cbv_design_matrix != []:
            # Combine cbv_design_matrix and extra_design_matrix 
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix], 
                                [self.extra_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        elif self.cbv_design_matrix != []:
            # Just use cbv_design_matrix 
            dm_to_flatten = [[cbv_dm for cbv_dm in self.cbv_design_matrix]]
            flattened_dm_list = [item for sublist in dm_to_flatten for item in sublist]
        else:
            # Just use extra_design_matrix
            flattened_dm_list = [self.extra_design_matrix]

        # Add in a constant to the design matrix collection
        # Note: correct_elasticnet ASSUMES the the last vector in the
        # design_matrix_collection is the constant
        flattened_dm_list.append(DesignMatrix(np.ones(flattened_dm_list[0].shape[0]),
            columns=['Constant'], name='Constant'))

        self.design_matrix_collection = DesignMatrixCollection(flattened_dm_list)


    def _set_prior_width(self, sigma):
        """ Sets the Gaussian prior in the design_matrix_collection widths to sigma

        Parameters
        ----------
        sigma : scalar float 
            all widths are set to the same value
            If sigma = None then uniform sigma is set
        """

        if (isinstance(sigma, list)):
            raise Exception("separate widths is not yet implemented")

        for dm in self.design_matrix_collection:
            nCBVs = len(dm.prior_sigma)

            if sigma is None:
                dm.prior_sigma = np.ones(nCBVs) * np.inf
            else:
                dm.prior_sigma = np.ones(nCBVs) * sigma


    def _goodness_metric_obj_fun(self, alpha):
        """ The objective function to minimize with
        scipy.optimize.minimize_scalar

        First sets the alpha regularization penalty then runs
        RegressionCorrector.correct and then computes the over- and
        under-fitting goodness metrics to return a scalar penalty term to
        minimize.

        Uses the paramaters in self.optimization_params.

        Parameters (in self.optimization_params)
        ----------
        alpha : float
            regularization penalty term value to set
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.
        target_over_score : float
            Target Over-fitting metric score
            If <=0 then ignore over-fitting metric
        target_under_score : float
            Target under-fitting metric score
            If <=0 then ignore under-fitting metric

        Returns
        -------
        penalty : float
            Penalty term for minimizer, based on goodness metrics
        """

        # Add in a width to the Gaussian priors
        # alpha = flux_sigma^2 / sigma^2
        sigma = np.median(self.lc.flux_err.value) / np.sqrt(np.abs(alpha))
        self._set_prior_width(sigma)                
        # Use RegressionCorrector.correct for the actual fitting
        self.correct_regressioncorrector(self.design_matrix_collection, 
            cadence_mask=self.optimization_params['cadence_mask'])

        # Do not compute and ignore if target score < 0
        if (self.optimization_params['target_over_score'] > 0):
            overMetric = self.over_fitting_metric(
                n_samples=self.optimization_params['over_metric_nSamples'])
        else: 
            overMetric = 1.0

        # Do not compute and ignore if target score < 0
        if (self.optimization_params['target_under_score'] > 0):
            underMetric = self.under_fitting_metric()
        else: 
            underMetric = 1.0

        # Once we hit the target we want to ease-back on increasing the metric
        # However, we don't want to ease-back to zero pressure, that will
        # unconstrain the penalty term and cause the optmizer to run wild.
        # So, use a "Leaky ReLU"
        # metric' = threshold + (metric - threshold) * leakFactor
        leakFactor = 0.01
        if (self.optimization_params['target_over_score'] > 0 and
                overMetric >= self.optimization_params['target_over_score']):
            overMetric = (self.optimization_params['target_over_score'] +
                leakFactor * 
                    (overMetric -
                        self.optimization_params['target_over_score']))

        if (self.optimization_params['target_under_score'] > 0 and
                underMetric >= self.optimization_params['target_under_score']):
            underMetric = (self.optimization_params['target_under_score'] +
                leakFactor * 
                    (underMetric -
                        self.optimization_params['target_under_score']))

        penalty = -(overMetric + underMetric)

        return penalty

    def diagnose(self):
        """ Returns diagnostic plots to assess the most recent correction.

        If a correction has not yet been fitted, a ``ValueError`` will be raised.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        axs = self._diagnostic_plot()

        plt.title('Alpha = {0:2.3e}'.format(self.alpha))
        
        return axs

    def goodness_metric_scan_plot(self, cbv_type=['SingleScale'],
            cbv_indices=[np.arange(1,9)], alpha_range_log10=[-4, 4],
            ext_dm=None, cadence_mask=None):
        """ Returns a diagnostic plot of the over and under goodness metrics as a
        function of the L2-Norm regularization term, alpha.

        alpha is scanned by default to the range 10^-4 : 10^4 in logspace

        cbvCorrector.correct_gaussian_prior is used to make the correction for
        each alpha. Then the over and under goodness metric are computed.

        If a correction has already been performed (via one of the correct_*
        methods) then the used alpha value is also plotted for reference.

        Parameters
        ----------
        cbv_type : str list
            List of CBV types to use in correction {'ALL' => Use all}
        cbv_indices : list of lists
            List of CBV vectors to use in each of cbv_type passed. {'ALL' => Use all}
            NOTE: 1-Based indexing!
        alpha_range_log10 : [list of two] The start and end exponent for the logspace scan.
            Default = [-4, 4]
        ext_dm  :  `.DesignMatrix` or `.DesignMatrixCollection`
            Optionally pass an extra design matrix to also be used in the fit
        cadence_mask : np.ndarray of bools (optional)
            Mask, where True indicates a cadence that should be used.

        Returns
        -------
        `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        alphaArray = np.logspace(alpha_range_log10[0], alpha_range_log10[1], num=100)

        # We need to make a copy of self so that the scan's final fit parameters
        # do not over-write any stored fit parameters
        cbvCorrectorCopy = self.copy()

        # Compute both metrics vs. alpha
        overMetric = []
        underMetric = []
        for thisAlpha in alphaArray:
            cbvCorrectorCopy.correct_gaussian_prior(cbv_type=cbv_type, cbv_indices=cbv_indices, 
                                        alpha=thisAlpha, ext_dm=ext_dm,
                                        cadence_mask=cadence_mask)
            overMetric.append(cbvCorrectorCopy.over_fitting_metric(n_samples=1))
            underMetric.append(cbvCorrectorCopy.under_fitting_metric())

        # plot both
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.semilogx(alphaArray, underMetric, 'b.', label='UnderFit')
        ax.semilogx(alphaArray, overMetric, 'r.', label='OverFit')

        if (isinstance(self.alpha, float)):
            ax.semilogx([self.alpha, self.alpha], [0, 1.0], 'k-', 
                label='corrected_lc Alpha = {0:2.3e}'.format(self.alpha))


        plt.title('Goodness Metrics vs. L2-Norm Penalty (alpha)')
        plt.xlabel('Regularization Factor Alpha')
        plt.ylabel('Goodness Metric')
        ax.grid(':', alpha=0.3)
        ax.legend()

        return ax

    def copy(self):
        """Returns a copy of this `cbvCorrector` object.

        This method uses Python's `copy.deepcopy` function to ensure that all
        objects stored within the cbvCorrector instance are fully copied.

        Returns
        -------
        cbvCorrector_copy : `cbvCorrector`
            A new object which is a copy of the original.
        """
        return copy.deepcopy(self)

    def __repr__(self):
        """ This will print all attributes of the class kinda like in
        self.__dict__
        """
        dictionary = self.__dict__.copy()

        dictionary['lc'] = '<{} targetid={} length={}>'.format(type(self.lc),
                self.lc.targetid, len(self.lc))
        if self.corrected_lc is not None:
            dictionary['corrected_lc'] = '<{} targetid={} length={}>'.format(
                    type(self.corrected_lc), self.corrected_lc.targetid, 
                    len(self.corrected_lc))
        
        dict_string = '\n'
        for key in dictionary.keys():
            dict_string += '\t{} = {}\n'.format(key, dictionary[key])
        
        return dict_string

#*******************************************************************************
#*******************************************************************************
#*******************************************************************************
# Cotrending Basis Vectors Classes and Functions 
#*******************************************************************************
#*******************************************************************************
#*******************************************************************************

class CotrendingBasisVectors(TimeSeries):
    """
    Defines a CotrendingBasisVectors class, which is the Superclass for
    KeplerCotrendingBasisVectors and TessCotrendingBasisVectors.
    Normally, one would use these latter classes instead of instantiating
    CotrendingBasisVectors directly. However, for generating custom CBVs one can
    use this super class.

    Stores Cotrending Basis Vectors for the Kepler/K2/TESS missions.

    Each CotrendingBasisVectors object contains only ONE set of CBVs.
    Instantiate multiple objects to store multiple set of CBVS, for example, to
    save each of the three multi-scale bands in TESS.

    CotrendingBasisVectors calls the standard __init__ from
    astropy.timeseries.TimeSeries

    Parameters
    ----------
    data : `~astropy.table.Table`
        Data to initialize CotrendingBasisVectors. The
        CBVs should be in columns called ``'CADENCENO'``, ``'GAP'``, ``'VECTOR_1'``,
        ``'VECTOR_2'``, ... ``'VECTOR_N'``
        If 'GAP' is not given then it is filled with all False.
        If 'CADENCENO' is not given then it is filled with np.arange(nCadences)
    time : `~astropy.time.Time`
        Time values.
    **kwargs : dict
        Additional keyword arguments are passed to `~astropy.table.QTable`.

    Attributes
    ----------
    cadenceno       : int array-like
        Cadence indices
    time            : flaot array-like
        CBV cadence times
    gap_indicators  : bool array-like
        True => cadence is gapped
    cbv_indices     : list int-like
        List of CBV indices available
        1-based indexing
    ['VECTOR_#']    : astropy.table.column.Column
        CBV number #

    """

    #***
    def __init__(self, data=None, time=None, **kwargs):

        # Add some columns if not existant
        if data is not None:
            if not 'GAP' in data.colnames:
                data['GAP'] = np.full(data[data.colnames[0]].size, False)
            if not 'CADENCENO' in data.colnames:
                data['CADENCENO'] = np.arange(data[data.colnames[0]].size)

        # Initialize the astropy.timeseries.TimeSeries attributes
        super().__init__(data=data, time=time, **kwargs)

        # Ensure all columns are Quantity objects
        for col in self.columns:
            if not isinstance(self[col], (Quantity, Time)):
                self.replace_column(col, Quantity(self[col], dtype=self[col].dtype))


    # cbv_indices are always determined by the 'VECTOR_#' columns in the
    # TimeSeries
    @property
    def cbv_indices(self):
        cbv_indices = []
        for name in self.colnames:
            if name.find('VECTOR_') > -1:
                cbv_indices.append(int(name[7:]))
        return cbv_indices

    @property
    def time(self):
        """The time values."""
        return self['time']

    @time.setter
    def time(self, time):
        self['time'] = time

    @property
    def gap_indicators(self):
        return self['GAP']

    @gap_indicators.setter
    def gap_indicators(self, gap_indicators):
        self['GAP'] = gap_indicators

    @property
    def cadenceno(self):
        return self['CADENCENO']

    @cadenceno.setter
    def cadenceno(self, cadenceno):
        self['CADENCENO'] = cadenceno

    def to_designmatrix(self, cbv_indices='all', name='CBVs'):
        """Returns a `DesignMatrix` where the columns are the
        requested CBVs.

        Parameters
        ----------
        cbv_indices : list of ints
            List of CBV vectors to use. 1-based indexing!
            {'all' => Use all}
        name : str
            A Name for the DesignMatrix

        Returns
        -------
            design_matrix : designmatrix.DesignMatrix
        """

        if isinstance(cbv_indices, str) and not cbv_indices == 'all':
            raise ValueError('cbv_indices must either be list of ints or "all"')
        elif not isinstance(cbv_indices, str) and 0 in cbv_indices:
            raise ValueError("CBVs use 1-based indexing. Do not request CBV index '0'")

        if (isinstance(cbv_indices, str) and (cbv_indices == 'all')):
            cbv_indices = self.cbv_indices

        cbv_names = []
        cbv_matrix = np.array([])
        for idx in cbv_indices:
            # Check that the CBV index is available
            if idx in self.cbv_indices:
                # If so, append it as a column to the matrix
                if len(cbv_matrix) == 0:
                    cbv_matrix =  np.array(self['VECTOR_{}'.format(idx)])[...,None]
                else:
                    cbv_matrix = np.hstack((cbv_matrix,
                        np.array(self['VECTOR_{}'.format(idx)])[...,None]))
                cbv_names.append('VECTOR_{}'.format(idx))

        return DesignMatrix(cbv_matrix, columns=cbv_names, name=name)

    def plot(self, cbv_indices='all', ax=None, **kwargs):
        """Plots the requested CBVs evenly spaced out vertically for legibility.

        Does not plot gapped cadences

        Parameters
        ----------
        cbv_indices : list of ints
            The list of cotrending basis vectors to plot. For example:
            [1, 2] will fit the first two basis vectors. 'all' => plot all
            NOTE: 1-based indexing
        ax : matplotlib.pyplot.Axes.AxesSubplot
            Matplotlib axis object. If `None`, one will be generated.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.pyplot.Axes.AxesSubplot
            Matplotlib axis object
        """

        if isinstance(cbv_indices, str) and not cbv_indices == 'all':
            raise ValueError('cbv_indices must either be list of ints or "all"')
        elif not isinstance(cbv_indices, str) and 0 in cbv_indices:
            raise ValueError("CBVs use 1-based indexing. Do not request CBV index '0'")


        with plt.style.context(MPLSTYLE):
            if (isinstance(cbv_indices, str) and (cbv_indices == 'all')):
                cbv_indices = []
                for name in self.colnames:
                    if name.find('VECTOR_') > -1:
                        cbv_indices.append(int(name[7:]))

            cbv_designmatrix = self.to_designmatrix(cbv_indices)

            if ax is None:
                _, ax = plt.subplots(1)

            # Plot gaps as NaN
            # time array is a Masked array so need to fill masks with nans
            timeArray = self.time.copy().value
            if isinstance(timeArray, (Masked, np.ma.MaskedArray)):
                if np.issubdtype(timeArray.dtype, np.int_):
                    timeArray = timeArray.astype(float)
                timeArray = timeArray.filled(np.nan)
            timeArray[np.nonzero(self.gap_indicators)[0]] = np.nan

            # Get the CBV arrays that were requested
            for idx, cbv_name in enumerate(cbv_designmatrix.columns):
                cbvIndex = cbv_name[7:]
                cbv = cbv_designmatrix[cbv_name]
                # Plot gaps as NaN
                cbv[np.nonzero(self.gap_indicators)[0]] = np.nan
                ax.plot(timeArray, cbv-idx/10., label='{}'.format(cbvIndex), **kwargs)

            ax.set_yticks([])
            ax.set_xlabel('Time [{}]'.format(self['time'].format))

            if hasattr(self, 'mission'):
                if self.mission == 'Kepler':
                    ax.set_title('Kepler CBVs (Quarter.Module.Output : {}.{}.{})'
                                 ''.format(self.quarter, self.module, self.output),
                                 fontdict={'fontsize': 10})
                elif self.mission == 'K2':
                    ax.set_title('K2 CBVs (Campaign.Module.Output : {}.{}.{})'
                                 ''.format( self.campaign, self.module, self.output),
                                 fontdict={'fontsize': 10})
                elif self.mission == 'TESS':
                    if (self.cbv_type == 'MultiScale'):
                        ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType.Band : {}.{})'
                                 ''.format(self.sector, self.camera, self.ccd, self.cbv_type, self.band),
                                 fontdict={'fontsize': 9})
                    else:
                        ax.set_title('TESS CBVs (Sector.Camera.CCD : {}.{}.{}, CBVType : {})'
                                 ''.format(self.sector, self.camera, self.ccd, self.cbv_type),
                                 fontdict={'fontsize': 10})
            else:
                # This is a generic CotrendingBasisVectors object
                ax.set_title('CBVs', fontdict={'fontsize': 10})

            ax.grid(':', alpha=0.3)
            ax.legend(fontsize='small', ncol=2)
        return ax

    def align(self, lc):
        """Aligns the CBVs to a light curve. The lightCurve object might not
        have the same cadences as the CBVs. This will trim the CBVs to be
        aligned with the light curve.

        This method will use the cadence number (lc.cadenceno) to
        perform the synchronization. Only cadence numbers that exist in both
        the CBVs and the light curve will have values in the returned CBVs. All
        cadence numbers that exist in the light curve but not in the CBVs will
        have NaNs returned for the CBVs on those cadences and the GAP set to
        True.

        Any cadences in the CBVs not in the light curve will be removed from the CBVs.

        The returned cbvs object is sorted by cadenceno.

        If you wish to interpolate the CBVs to arbitrary light curve cadence
        times then use the interpolate method.

        Parameters
        ----------
        lc : LightCurve object
            The reference light curve to align to

        Returns
        -------
        cbvs : CotrendingBasisVectors object
            Aligned to the light curve
        """

        # The fraction of cadences that do not align to throw a
        # warning about the CBVs being poorly aligned to the light curve
        poorly_aligned_threshold = 0.5
        poorly_aligned_flag = False

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        if hasattr(lc, 'cadenceno'):

            # Make a deepcopy so we do not just return a modified original
            cbvs = copy.deepcopy(self)

            # NaN any CBV cadences that are in the light curve and not in CBVs
            # This requires us to add rows to the CBV table
            lc_nan_mask = np.logical_not(np.in1d(lc.cadenceno, cbvs.cadenceno))
            # Determine if the CBVs are poorly aligned to the light curve
            if ((np.count_nonzero(lc_nan_mask) / len(lc_nan_mask)) >
                            poorly_aligned_threshold):
                poorly_aligned_flag = True

            lc_nan_indices = np.nonzero(lc_nan_mask)[0]
            # Sadly, there is no TimesSeries.add_rows (plural), so we have to
            # add each row in a for-loop
            if len(lc_nan_indices) > 0:
                for idx in lc_nan_indices:
                    dict_to_add = {}
                    dict_to_add['time'] = lc.time[idx]
                    dict_to_add['CADENCENO'] = lc.cadenceno[idx]
                    dict_to_add['GAP'] = True
                    for cbvIdx in cbvs.cbv_indices:
                        dict_to_add['VECTOR_{}'.format(cbvIdx)] = np.nan

                    cbvs.add_row(dict_to_add)

            # There appears to be a bug in astropy.timeseries when using ts[x:y]
            # in combination with ts.remove_row() or ts.remove_rows.
            # See LightKurve Issue #836.
            # To get around the error for now, we will attempt to use
            # ts[x:y]. If it errors out then revert to remove_rows, which is
            # REALLY slow.
            try:
                # This method is fast but might cause errors
                keep_indices = np.nonzero(np.in1d(cbvs.cadenceno, lc.cadenceno))[0]
                # Determine if the CBVs are poorly aligned to the light curve
                if (len(keep_indices) / len(cbvs)) < poorly_aligned_threshold:
                    poorly_aligned_flag = True
                cbvs = cbvs[keep_indices]
            except:
                # This method is slow but appears to be more robust
                trim_indices = np.nonzero(np.logical_not(
                    np.in1d(cbvs.cadenceno, lc.cadenceno)))[0]
                # Determine if the CBVs are poorly aligned to the light curve
                if (len(trim_indices) / len(cbvs)) > poorly_aligned_threshold:
                    poorly_aligned_flag = True
                cbvs.remove_rows(trim_indices)

            # Now sort the CBVs by cadenceno
            cbvs.sort('CADENCENO')

        else:
            raise Exception('align requires cadence numbers for the ' + \
                    'light curve. NO SYNCHRONIZATION OCCURED')

        # Only issue this warning once
        if poorly_aligned_flag:
            log.warning('The {} CBVs do not appear to be well aligned to the '
                'light curve. Consider using "interpolate_cbvs=True"'.format(cbvs.cbv_type))

        return cbvs

    def interpolate(self, lc, extrapolate=False):
        """Interpolates the CBV to the cadence times in the given light curve
        using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP).

        Uses scipy.interpolate.PchipInterpolator

        Each CBV is interpolated independently. All gaps are set to False.
        The cadence numbers are taken from the light curve.

        Parameters
        ----------
        lc : LightCurve object
            The reference light curve cadence times to interpolate to
        extrapolate : bool, optional
            Whether to extrapolate to out-of-bounds points based on first
            and last intervals, or to return NaNs.

        Returns
        -------
        cbvs_interpolated: CotrendingBasisVectors object
            interpolated to the light curve cadence times
        """

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        # If not extrapolating then check if extrapolation is necessary.
        # If so, throw a warning
        if extrapolate==False:
            gapRemovedCBVtime = self.time.value[np.logical_not(self.gap_indicators.value)]
            if (np.min(lc.time.value) < np.min(gapRemovedCBVtime) or
                np.max(lc.time.value) > np.max(gapRemovedCBVtime)   ):
                log.warning('Extrapolation of CBVs appears to be necessary. '
                            'Extrapolated values will be filled with zeros. '
                            'Recommend setting extrapolate=True')

        # Create the new cbv object with no basis vectors, yet...
        cbvNewTime = lc.time.copy()
        # Gaps are all false
        gaps = np.full(len(lc.time), False)
        dataTbl = Table([lc.cadenceno, gaps], names=('CADENCENO', 'GAP'))

        # We are PCHIP interpolating each CBV independently.
        # Do not include gaps when interpolating
        warning_posted = False
        for idx in self.cbv_indices:
            fInterp = PchipInterpolator(
                    self.time.value[np.logical_not(self.gap_indicators.value)],
                    self['VECTOR_{}'.format(idx)][np.logical_not(self.gap_indicators.value)], 
                    extrapolate=extrapolate)
            dataTbl['VECTOR_{}'.format(idx)] = fInterp(lc.time.value)
            # Replace NaNs with 0.0
            if (np.any(np.isnan(dataTbl['VECTOR_{}'.format(idx)]))):
                dataTbl['VECTOR_{}'.format(idx)][np.isnan(dataTbl['VECTOR_{}'.format(idx)])] = \
                    np.full(np.count_nonzero(np.isnan(dataTbl['VECTOR_{}'.format(idx)])), 0.0)
                # Only post this warning once
                if (not warning_posted):
                    log.warning('Some interpolated (or extrapolated) CBV values have been set to zero')
                    warning_posted = True

        dataTbl.meta = self.meta.copy()

        # We need to return a new CotrendingBasisVectors class. Make sure we
        # instantiate the correct type.
        if isinstance(self, KeplerCotrendingBasisVectors):
            return KeplerCotrendingBasisVectors(data=dataTbl, time=cbvNewTime)
        elif isinstance(self, TessCotrendingBasisVectors):
            return TessCotrendingBasisVectors(data=dataTbl, time=cbvNewTime)
        else:
            return CotrendingBasisVectors(data=dataTbl, time=cbvNewTime)


class KeplerCotrendingBasisVectors(CotrendingBasisVectors):
    """Sub-class for Kepler/K2 cotrending basis vectors

    See CotrendingBasisVectors for class details

    Attributes
    ----------
    CotrendingBasisVectors attributes
    astropy.timeseries.TimeSeries attributes
    mission         : [str] ('Kepler', 'K2')
    cbv_type        : [str] always 'SingleScale'
    quarter         : [int] Kepler Quarter
    campaign        : [int] K2 Campaign
    module          : [int] Kepler instrument CCD module
    output          : [int] Kepler instrument CCD output

    """

    #***
    validMissionOptions = ('Kepler', 'K2')
    validCBVTypes = ('SingleScale')

    #***

    def __init__(self, data=None, time=None, **kwargs):
        """Initiates a KeplerCotrendingBasisVectors object.
        Normally one would use KeplerCotrendingBasisVectors.from_hdu to
        automatically set up the object. However, for certain functionality
        one must instantiate the object directly.
        """

        # Initialize attributes common to all CotrendingBasisVector classes
        super(KeplerCotrendingBasisVectors, self).__init__(data=data,
                time=time, **kwargs)

    @classmethod
    def from_hdu(self, hdu=None, module=None, output=None,
            **kwargs):
        """Class method to instantiate a KeplerCotrendingBasisVectors object
        from a CBV FITS HDU.

        Kepler/K2 CBVs are all in the same FITS file for each quarter/campaign,
        so, when instantiating the CBV object we must specify which module and
        output we desire. Only Single-Scale CBVs are stored for Kepler.

        Parameters
        ----------
        hdu : astropy.io.fits.hdu.hdulist.HDUList
            A pyfits opened FITS file containing the CBVs
        module : int
            Kepler CCD module 2 - 84
        output : int
            Kepler CCD output 1 - 4
        **kwargs : Optional arguments
            Passed to the TimeSeries superclass
        """

        assert module > 1 and module < 85, 'Invalid module number'
        assert output > 0 and output < 5, 'Invalid output number'

        # Get the mission: Kepler or K2
        # Sadly, the HDU does not explicitly say if this is Kepler or K2 CBVs.
        if 'QUARTER' in hdu['PRIMARY'].header:
            mission = 'Kepler'
        elif 'CAMPAIGN' in hdu['PRIMARY'].header:
            mission = 'K2'
        else:
            raise Exception('This does not appear to be a Kepler or K2 FITS HDU')

        extName = 'MODOUT_{0}_{1}'.format(module, output)

        try:
            # Read the columns and meta data
            with warnings.catch_warnings():
                # By default, AstroPy emits noisy warnings about units commonly used
                # in archived TESS data products (e.g., "e-/s" and "pixels").
                # We ignore them here because they don't affect Lightkurve's features.
                # Inconsistencies between TESS data products and the FITS standard
                # out to be addressed at the archive level. (See issue #1216.)
                warnings.simplefilter("ignore", category=UnitsWarning)
                dataTbl = Table.read(hdu[extName], format="fits")
            dataTbl.meta.update(hdu[0].header)
            dataTbl.meta.update(hdu[extName].header)

            # TimeSeries-based objects require a dedicated time column
            # Replace NaNs with default time '2000-01-01', otherwise,
            # astropy.time.Time complains
            nanHere = np.nonzero(np.isnan(dataTbl['TIME_MJD'].data))[0]
            timeData = dataTbl['TIME_MJD'].data
            timeData[nanHere] = Time(['2000-01-01'], scale='utc').mjd
            cbvTime = Time(timeData, format='mjd', scale='utc')
            dataTbl.remove_column('TIME_MJD')

            # Gaps are labelled as 'GAPFLAG' so rename!
            dataTbl['GAP'] = dataTbl['GAPFLAG']
            dataTbl.remove_column('GAPFLAG')

            dataTbl.meta['MISSION'] = mission
            dataTbl.meta['CBV_TYPE'] = 'SingleScale'

        except:
            dataTbl = None
            cbvTime = None

        # Here we instantiate the actual object
        return self(data=dataTbl, time=cbvTime, **kwargs)

    @property
    def mission(self):
        return self.meta.get('MISSION', None)

    @mission.setter
    def mission(self, mission):
        self.meta['MISSION'] = mission

    @property
    def cbv_type(self):
        return self.meta.get('CBV_TYPE', None)

    @cbv_type.setter
    def cbv_type(self, cbv_type):
        self.meta['CBV_TYPE'] = cbv_type

    @property
    def quarter(self):
        return self.meta.get('QUARTER', None)

    @quarter.setter
    def quarter(self, quarter):
        if (self.mission == 'Kepler'):
            self.meta['QUARTER'] = quarter
        else:
            pass

    @property
    def campaign(self):
        return self.meta.get('CAMPAIGN', None)

    @campaign.setter
    def campaign(self, campaign):
        if (self.mission == 'K2'):
            self.meta['CAMPAIGN'] = campaign
        else:
            pass

    @property
    def module(self):
        return self.meta.get('MODULE', None)

    @module.setter
    def module(self, module):
        self.meta['MODULE'] = module

    @property
    def output(self):
        return self.meta.get('OUTPUT', None)

    @output.setter
    def output(self, output):
        self.meta['OUTPUT'] = output

    def __repr__(self):

        if self.mission == 'Kepler':
            repr_string = 'Kepler CBVs, Quarter.Module.Output : {}.{}.{}, nCBVs : {}'\
                ''.format(self.quarter, self.module, self.output, len(self.cbv_indices))
        elif self.mission == 'K2':
            repr_string = 'K2 CBVs, Campaign.Module.Output : {}.{}.{}, nCBVs : {}'\
                ''.format( self.campaign, self.module, self.output, len(self.cbv_indices))

        return repr_string


class TessCotrendingBasisVectors(CotrendingBasisVectors):
    """ Sub-class for TESS cotrending basis vectors

    See CotrendingBasisVectors for class details

    Attributes
    ----------
    CotrendingBasisVectors attributes
    astropy.timeseries.TimeSeries attributes
    mission         : [str] ('TESS')
    cbv_type        : [str ('SingleScale', 'MultiScale', 'Spike')
    sector          : [int] TESS Sector
    camera          : [int] TESS Camera Index
    ccd             : [int] TESS CCD Index
    band            : [int] MultiScale band number (invalid for other CBV types)

    """

    validMissionOptions = ('TESS')
    validCBVTypes = ('SingleScale', 'MultiScale', 'Spike')

    def __init__(self, data=None, time=None, **kwargs):
        """Initiates a TessCotrendingBasisVectors object.

        Normally one would use TessCotrendingBasisVectors.from_hdu to
        automatically set up the object. However, for certain functionaility
        one must instantiate the object directly.
        """

        # Initialize attributes common to all CotrendingBasisVector classes
        super(TessCotrendingBasisVectors, self).__init__(data=data,
                time=time, **kwargs)

    @classmethod
    def from_hdu(self, hdu=None, cbv_type=None, band=None, **kwargs):
        """Class method to instantiate a TessCotrendingBasisVectors object
        from a CBV FITS HDU.

        TESS CBVs are in separate FITS files for each camera.CCD, so camera.CCD
        is already specified in the HDU, here we need to specify
        which CBV type and band is desired.

        If the requested CBV type does not exist in the HDU then None is
        returned

        Parameters
        ----------
        hdu : astropy.io.fits.hdu.hdulist.HDUList
            A pyfits opened FITS file containing the CBVs
        cbv_type : str
            'SingleScale', 'MultiScale' or 'Spike'
        band : int
            Band number for 'MultiScale' CBVs
            Ignored for 'SingleScale' or 'Spike'
        **kwargs : Optional arguments
            Passed to the TimeSeries superclass
        """

        mission = hdu['PRIMARY'].header['TELESCOP']
        assert mission == 'TESS', 'This does not appear to be a TESS FITS HDU'

        # Check if a valid cbv_type and band was passed
        if not cbv_type in self.validCBVTypes:
            raise ValueError('Invalid cbv_type')
        if band is not None and band < 1:
            raise ValueError('Invalid band')

        # Get the requested cbv_type
        # Curiosly, camera and CCD are not in the primary header!
        camera = hdu[1].header['CAMERA']
        ccd = hdu[1].header['CCD']
        switcher = {
            'SingleScale': 'CBV.single-scale.{}.{}'.format(camera, ccd),
            'MultiScale': 'CBV.multiscale-band-{}.{}.{}'.format(band,
                camera, ccd),
            'Spike': 'CBV.spike.{}.{}'.format(camera, ccd),
            'unknown': 'error'
            }
        extName = switcher.get(cbv_type, switcher['unknown'])
        if (extName == 'error'):
            raise Exception('Invalide cbv_type')

        try:

            # Read the columns and meta data
            with warnings.catch_warnings():
                # By default, AstroPy emits noisy warnings about units commonly used
                # in archived TESS data products (e.g., "e-/s" and "pixels").
                # We ignore them here because they don't affect Lightkurve's features.
                # Inconsistencies between TESS data products and the FITS standard
                # out to be addressed at the archive level. (See issue #1216.)
                warnings.simplefilter("ignore", category=UnitsWarning)
                dataTbl = Table.read(hdu[extName], format="fits")
            dataTbl.meta.update(hdu[0].header)
            dataTbl.meta.update(hdu[extName].header)

            # TimeSeries-based objects require a dedicated time column
            # Replace NaNs with default time '2000-01-01', otherwise,
            # astropy.time.Time complains
            nanHere = np.nonzero(np.isnan(dataTbl['TIME'].data))[0]
            timeData = dataTbl['TIME'].data
            timeData[nanHere] = Time(['2000-01-01'], scale='tdb').mjd
            cbvTime = Time(timeData, format='btjd', scale='tdb')
            dataTbl.remove_column('TIME')

            dataTbl.meta['MISSION'] = 'TESS'
            dataTbl.meta['CBV_TYPE'] = cbv_type
            dataTbl.meta['BAND'] = band

        except:
            dataTbl = None
            cbvTime = None

        # Here we instantiate the actual object
        return self(data=dataTbl, time=cbvTime, **kwargs)

    @property
    def mission(self):
        return self.meta.get('MISSION', None)

    @mission.setter
    def mission(self, mission):
        self.meta['MISSION'] = mission

    @property
    def cbv_type(self):
        return self.meta.get('CBV_TYPE', None)

    @cbv_type.setter
    def cbv_type(self, cbv_type):
        self.meta['CBV_TYPE'] = cbv_type

    @property
    def band(self):
        return self.meta.get('BAND', None)

    @band.setter
    def band(self, band):
        self.meta['BAND'] = band

    @property
    def sector(self):
        return self.meta.get('SECTOR', None)

    @sector.setter
    def sector(self, sector):
        self.meta['SECTOR'] = sector

    @property
    def camera(self):
        return self.meta.get('CAMERA', None)

    @camera.setter
    def camera(self, camera):
        self.meta['CAMERA'] = camera

    @property
    def ccd(self):
        return self.meta.get('CCD', None)

    @ccd.setter
    def ccd(self, ccd):
        self.meta['CCD'] = ccd

    def __repr__(self):

        if (self.cbv_type == 'MultiScale'):
            repr_string = 'TESS CBVs, Sector.Camera.CCD : {}.{}.{}, CBVType.Band: {}.{}, nCBVs : {}' \
                ''.format(self.sector, self.camera, self.ccd, self.cbv_type,
                    self.band, len(self.cbv_indices))
        else:
            repr_string = 'TESS CBVs, Sector.Camera.CCD : {}.{}.{}, CBVType : {}, nCBVS : {}'\
                ''.format(self.sector, self.camera, self.ccd, self.cbv_type, len(self.cbv_indices))

        return repr_string

#*******************************************************************************
# Functions






@deprecated("2.1", alternative="load_kepler_cbvs", warning_type=LightkurveDeprecationWarning)
def download_kepler_cbvs(*args, **kwargs):
    return load_kepler_cbvs(*args, **kwargs)


def load_kepler_cbvs(cbv_dir=None,mission=None, quarter=None, campaign=None,
        channel=None, module=None, output=None):
    """Loads Kepler or K2 cotrending basis vectors, either from a local directory cbv_dir 
    or searches the public data archive at MAST <https://archive.stsci.edu>.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    mission, quarter/campaign and channel or module/output, etc...
    and then extracts the requested basis vectors and returns a
    KeplerCotrendingBasisVectors object

    For Kepler/K2, the FITS files contain all channels in a single file per
    quarter/campaign.

    For Kepler this extracts the DR25 CBVs.

    Parameters
    ----------
    cbv_dir : str
        Path to specific directory holding Kepler CBVs. If None, queries MAST.
    mission : str, list of str
        'Kepler' or 'K2'
    quarter or campaign : int
        Kepler Quarter or K2 Campaign.
    channel or (module and output) : int
        Kepler/K2 requested channel or module and output.
        Must provide either channel, or module and output,
        but not both.

    Returns
    -------
    result : :class:`KeplerCotrendingBasisVectors` object

    Examples
    --------
    This example will read in the CBVs for Kepler quarter 8,
    and then extract the first 8 CBVs for module.output 16.4

        >>> cbvs = load_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4) # doctest: +SKIP

    """

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    if (mission == 'Kepler'):
        assert  isinstance(quarter, int), 'quarter must be passed for Kepler mission'
        assert  campaign is None, 'campaign must not be passed for Kepler mission'
    elif (mission == 'K2'):
        assert  isinstance(campaign, int), 'campaign must be passed for K2 mission'
        assert  quarter is None,  'quarter must not be passed for K2 mission'
    else:
        raise ValueError('Unknown mission type')

    # CBV FITS files use module/output, not channel
    # So if channel is passed, convert to module/output
    if (isinstance(channel, int)):
        assert  module is None, 'module must NOT be passed if channel is passed'
        assert  output is None, 'output must NOT be passed if channel is passed'
        module, output = channel_to_module_output(channel)
        channel = None
    else:
        assert  module is not None, 'module must be passed'
        assert  output is not None, 'output must be passed'

    if cbv_dir:
        cbvBaseUrl = ""
    elif (mission == 'Kepler'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
    elif (mission == 'K2'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"

    try:
        kepler_cbv_fname = None
        if cbv_dir:
            cbv_files = glob.glob(os.path.join(cbv_dir,'*.fits'))
        else:
            soup = BeautifulSoup(requests.get(cbvBaseUrl).text, 'html.parser')
            cbv_files = [fn['href'] for fn in soup.find_all('a') if fn['href'].endswith('fits')]

        if mission == 'Kepler':
            quarter = 'q{:02}'.format(quarter)
            for cbv_file in cbv_files:
                if quarter + '-d25' in cbv_file:
                    break
        elif mission == 'K2':
            campaign = 'c{:02}'.format(campaign)
            for cbv_file in cbv_files:
                if campaign in cbv_file:
                    break

        kepler_cbv_fname = cbvBaseUrl + cbv_file
        hdu = pyfits.open(kepler_cbv_fname)
        return KeplerCotrendingBasisVectors.from_hdu(hdu=hdu, module=module, output=output)

    except Exception as e:
        raise Exception('CBVS were not found') from e


@deprecated("2.1", alternative="load_tess_cbvs", warning_type=LightkurveDeprecationWarning)
def download_tess_cbvs(*args, **kwargs):
    return load_tess_cbvs(*args, **kwargs)


def load_tess_cbvs(cbv_dir=None,sector=None, camera=None,
        ccd=None, cbv_type='SingleScale', band=None):
    """Loads TESS cotrending basis vectors, either from a directory of 
    CBV files already saved locally if cbv_dir is passed, or else 
    will retrieve the relevant files programmatically from MAST. 

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    cotrending basis vectors.

    For TESS, each CCD CBVs are stored in a separate FITS files.

    For now, this function will only load 2-minute cadence CBVs. Once other
    cadence CBVs become available this function will be updated to support
    their downloads.

    Parameters
    ----------
    cbv_dir   : str
        Path to specific directory holding TESS CBVs. If None, queries MAST.
    sector : int, list of ints
        TESS Sector number.
    camera and ccd : int
        TESS camera and CCD
    cbv_type : str
        'SingleScale' or 'MultiScale' or 'Spike'
    band : int
        Multi-scale band number

    Returns
    -------
    result : :class:`TessCotrendingBasisVectors` object

    Examples
    --------
    This example will load presaved CBVs from directory '.' for TESS Sector 10 Camera.CCD 2.4
    Multi-Scale band 2

        >>> cbvs = load_tess_cbvs('.',sector=10, camera=2, ccd=4, # doctest: +SKIP
        >>>     cbv_type='MultiScale', band=2) # doctest: +SKIP
    """

    # The easiest way to obtain a link to the CBV file for a TESS Sector and
    # camera.CCD is
    #
    # 1. Download the bulk download curl script (with a predictable url) for the
    # desired sector and search it for the camera.CCD needed
    # 2. Download the CBV FITS file based on the link in the curl script
    #
    # The bulk download curl links have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_17_cbv.sh
    #
    # Then the individual CBV files found in the curl file have urls such as:
    #
    # https://archive.stsci.edu/missions/tess/ffi/s0017/2019/279/1-1/tess2019279210107-s0017-1-1-0161-s_cbv.fits

    #***
    # Validate inputs
    # Make sure only the appropriate arguments are passed
    assert  isinstance(sector, int),    'sector must be passed for TESS mission'
    assert  isinstance(camera, int),    'camera must be passed'
    assert  isinstance(ccd, int),       'CCD must be passed'
    if cbv_type == 'MultiScale':
        assert  isinstance(band, int),  'band must be passed for multi-scale CBVs'
    else:
        assert  band is None,  'band must NOT be passed for single-scale or spike CBVs'

    # This is the string to search for in the curl script file
    # Pad the sector number with a first '0' if less than 10
    # TODO: figure out a way to pad an integer number with forward zeros
    # without needing a conditional
    sector = int(sector)

    try:
        SearchString = 's%04d-%s-%s-' % (sector, str(camera),str(ccd))
    except:
        raise Exception('Error parsing sector string when getting TESS CBV FITS files')

    try:
        if cbv_dir is not None:
            # Read in the relevant curl script file and find the line for the CBV
            # data we are looking for
            data = glob.glob(os.path.join(cbv_dir,'*.fits'))
            fname = None
            for line in data:
                strLine = str(line)
                if SearchString in strLine:
                    fname = strLine
                    break
            if (fname is None):
                raise Exception('CBV FITS file not found')

            # Extract url from strLine

            hdu = pyfits.open(fname)

        else:
            curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
            curlEndUrl = '_cbv.sh'
            curlUrl = curlBaseUrl + str(sector) + curlEndUrl

            # This is the string to search for in the curl script file

            # Read in the relevant curl script file and find the line for the CBV
            # data we are looking for
            data = urllib.request.urlopen(curlUrl)
            foundIndex = None
            for line in data:
                strLine = str(line)
                if SearchString in strLine:
                    foundIndex = strLine.index(SearchString)
                    break
            if (foundIndex is None):
                raise Exception('CBV FITS file not found')

            # Extract url from strLine
            htmlStartIndex = strLine.find('https:')
            htmlEndIndex = strLine.rfind('fits')
            # Add 4 for length of 'fits' string
            tess_cbv_url  = strLine[htmlStartIndex:htmlEndIndex+4]

            hdu = pyfits.open(tess_cbv_url)

        # Check that this is a TESS CBV FITS file
        mission = hdu['Primary'].header['TELESCOP']
        validate_method(mission, ['tess'])

        return TessCotrendingBasisVectors.from_hdu(hdu=hdu, cbv_type=cbv_type, band=band)

    except:
        raise Exception('CBVS were not found')
