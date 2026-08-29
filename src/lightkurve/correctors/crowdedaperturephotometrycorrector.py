"""Defines CrowdedAperturePhotometryCorrector
"""
import logging
import warnings

import numpy as np
import matplotlib
import lightkurve as lk
from lightkurve.correctors import PLDCorrector
from astropy import units as u

from .corrector import Corrector

from ..lightcurve import LightCurve, MPLSTYLE

__all__ = ['CrowdedAperturePhotometryCorrector']

log = logging.getLogger(__name__)


class CrowdedAperturePhotometryCorrector(Corrector):
    """Implements the crowding correction on a SAP light curve that has already 
    been corrected for instrumental noise/scattered light using a pervious 
    corrector function. 

     A description of this correction and its application is provided in Section 2.3.11 of this paper - https://iopscience.iop.org/article/10.1086/667698/pdf

    The crowding correction applied focuses on two paramters:
    *The crowding metric: This reflects what fraction of the flux in the 
    aperture is due to the target itself, not the nearby light sources.
    *The flux fraction: Similar to excess flux leaking into the aperture, 
    a fraction of the PSF of the target may not be captured in it. To account 
    for this missing fraction, the flux fraction is computed.

    The correction works by first calculating the median flux of the corrected 
    timeseries. Then calculating the excess flux via (1-CROWDSAP) * med_flux, 
    which is subtracted from the time series. Next you account for the flux of 
    the object outside of the aperture by dividing it by FLFRCSAP.

    The CROWDSAP and FLFRCSAP are keywords stored in the TPF. 
    Note that the light curve must be generated from the optimal aperture stored
    in the TPF as well. Otherwise the keywords are not applicable.

    Examples
    ------
    Download the TPF file for WR21a, obtain a PLD-corrected light curve, and
    then correct it for crowding:

    >>> import lightkurve as lk
    >>> from lightkurve.correctors import PLDCorrector
    >>> tpf = lk.search_targetpixelfile('WR21a', sector=36).download(quality_bitmask='hard')
    >>> tpf_lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    >>> pld = PLDCorrector(tpf, aperture_mask=tpf.pipeline_mask)
    >>> pld_lc = pld.correct(pca_components=5, aperture_mask=tpf.pipeline_mask)
    >>> crowdsap = tpf.hdu[1].header['CROWDSAP']
    >>> flfrsap = tpf.hdu[1].header['FLFRCSAP']
    >>> cc = CrowdedAperturePhotometryCorrector(pld_lc)
    >>> lc = cc.correct(crowdsap=crowdsap, flfrcsap=flfrsap)"""

    def __init__(self, lc):
        self.lc = lc
        self.flux = lc.flux
        self.flux_err = lc.flux_err
        self.time = lc.time
        self.meta = lc.meta

    def correct(self, crowdsap=None, flfrcsap=None):

        #Get the CROWDSAP function from the LCF
        if "CROWDSAP" in self.meta:
            crowdsap = self.meta.get("CROWDSAP")
        elif crowdsap !=None:
            crowdsap = float(crowdsap)
        else:
            crowdsap = None
            raise ValueError("crowdsap is not defined")
            

        #Get the FLFRCSAP function from the LCF
        if "FLFRCSAP" in self.meta:
            flfrcsap = self.meta.get("FLFRCSAP")
        elif flfrcsap !=None:
            flfrcsap = float(flfrcsap)
        else:
            flfrcsap = None
            raise ValueError("flfrcsap is not defined")
            

                
        #Get the median flux from the input light curve - this should be an SAP light curve
        #that has been corrected for scattered light and noise
        median_flux = np.nanmedian(self.flux.value)
            
        #Calculate the excess flux
        excess_flux = (1-crowdsap)*median_flux
        
        #Remove excess flux from the lc
        flux_removed = self.flux.value - excess_flux
        
        #Adjust for target flux outside the aperture
        flux_corr = flux_removed/flfrcsap
        
        #Calculate the new uncertainties
        flux_err_corr = self.flux_err.value/flfrcsap

        #Define units for the lc
        unit = self.flux.unit

        #Copy the original lc
        corrected_lc = self.lc.copy()

        #Adjust the flux and the flux error - this preserves meta data
        corrected_lc.flux =  flux_corr*unit
        corrected_lc.flux_err = flux_err_corr*unit
            
        return corrected_lc

    def diagnose(self):
        return None

    
