"""Defines the BiasBackgroundCorrector"""
import logging
import warnings

import numpy as np
import matplotlib
import lightkurve as lk
from scipy import ndimage

from .corrector import Corrector

from ..lightcurve import LightCurve, MPLSTYLE

__all__ = ['BiasBackgroundCorrector']

log = logging.getLogger(__name__)

class BiasBackgroundCorrector(Corrector):
    """During TESS's primary mission relatively dim and/or crowded target stars
    were often subject to overestimated background levels, which would result 
    in underestimated flux values and therefore overestimated transit depths.
    In Sector 27 a procedure was added to improve the accuracy of the 
    background correction. 

    This update has been applied to all 2-minute and  20-second targets. 
    Previous Sector data however, has not been corrected.
    While the change in the background estimates and relative transit depths 
    is generally small (< 2% for transit depths and < 1% for planet radii), 
    for some dim and/or crowded targets the effect is larger.

    This corrector class calculate the scaler offset (pdcCorrection) to apply 
    to the PDCSAP_FLUX in Years 1 and 2, to correct for the measured background
    bias . It also calculates the predicted bias (transitBias) in the transit 
    depth given the bias in the background estimate. This value is diagnostic 
    only, but useful in determining if the background bias is significant for a
    given target object.

    The correction works by first removing potentially problematic pixels from 
    the background pixel sample. It then calculates the median flux of 
    each remaining pixel in the TPF. 
    The median pixel values are then sorted and the one with the third lowest
    value is selected - this is bgBias value.

    Next the CROWSAP and FLFRCSAP keyvalues are used to calculate the pdc 
    correctiob  via pdcCorrection = bgBias*nPix*CROWDSAP/FLFRCSAP - where nPix
    is the number of pixels in the SPOC defined optimal aperture. This value 
    is added to the PDCSAP flux.

    Then the transitBias value is calculated via dividing the correction by 
    the median value of the PDCSAP_FLUX time series.

    Exaples
    ------
    Download the TPF file and the PDCSAP lightcurve for TOI-824
    then correct it for the background:

    >>> import lightkurve as lk
    >>> from scipy import ndimage
    >>> tpf = lk.search_targetpixelfile('TOI-824', sector=11).download()
    >>> lcf = lk.search_lightcurve('TOI-824', author='SPOC', sector=11).download()
    >>> pix = tpf.flux.value
    >>> inBackgroundAperture = ~tpf.pipeline_mask 
    >>> halo = ndimage.binary_dilation(tpf.pipeline_mask, iterations=1) 
    >>> inBackgroundAperture2 = inBackgroundAperture * ~halo
    >>> svalue = 200000
    >>> tp = np.where(tpf.flux.value >=svalue)
    >>> tp1 = tp[1]
    >>> tp2 = tp[2] 
    >>> sat_mask = np.zeros((tpf.shape[1:]), dtype='bool')
    >>> sat_mask[tp1,tp2] = True
    >>> aper_new2 = ndimage.binary_dilation(sat_mask, iterations=2) 
    >>> inBackgroundAperture3 = inBackgroundAperture2 * ~aper_new2
    >>> bkg = np.where(inBackgroundAperture3==True)
    >>> bkg2 = [bkg[0],bkg[1]]
    >>> medval = []
    >>> for a in range(tpf.shape[1:]):
    >>>
    >>>     custom_mask = np.zeros((tpf.shape[1:]), dtype='bool')
    >>>     custom_mask[bkg2[0][a],bkg2[1][a]] = True
    >>>     lc = tpf[tpf.quality == 0].to_lightcurve(aperture_mask=custom_mask)
    >>>     medval.append(np.nanmedian(lc.flux.value))
    >>> sorted_index = np.argsort(medval)
    >>> sort_int = [sorted_index[0],sorted_index[1],sorted_index[2]]
    >>> vals2 = [medval[sorted_index[0]],medval[sorted_index[1]],medval[sorted_index[2]]]
    >>> if vals2[2] < 0:
    >>>     bgBias = np.abs(vals2[2])
    >>> else:
    >>>     bgBias = 0
    >>> print("This is the estimate for the background bias:", bgBias)
    >>> nPix = np.sum(tpf.pipeline_mask)
    >>> CROWDSAP = tpf.hdu[1].header['CROWDSAP']
    >>> FLFRCSAP = tpf.hdu[1].header['FLFRCSAP']
    >>> medFluxPdc = np.nanmedian(lcf[lcf.quality == 0].flux.value)
    >>> pdcCorrection = bgBias*nPix*CROWDSAP/FLFRCSAP
    >>> cor_flux = lcf.flux.value + pdcCorrection
    >>> lc_corr = lk.LightCurve(time=lcf.time.value, flux=cor_flux, flux_err=lcf.flux_err.value)
    >>> transitBias = (pdcCorrection/medFluxPdc)*100
    >>> print("transitBias value:", transitBias, "%")"""

    
    def __init__(self, tpf, lc):
        self.tpf = tpf
        self.flux = tpf.flux
        self.flux_err = tpf.flux_err
        self.time = tpf.time
        self.quality = tpf.quality
        self.shape = tpf.shape
        self.pipeline_mask = tpf.pipeline_mask
        self.lc = lc
        self.lc.flux = lc.flux
        self.lc.time = lc.time
        self.lc.quality = lc.quality

    def correct(self):

        #Get the TPF optimal aperture mask and use its inverse to define the background
        inBackgroundAperture = ~self.pipeline_mask

        #Exclude pixels in optimal aperture + a 1 pixel halo around it.
        halo = ndimage.binary_dilation(self.pipeline_mask, iterations=1)
        inBackgroundAperture2 = inBackgroundAperture * ~halo

        #Remove any pixels from the background that might be saturated + a 2 pixel halo around them
        svalue = 200000 #This is an approximation only
        tp = np.where(self.flux.value >=svalue)
        
        tp1 = tp[1] #x-axis
        tp2 = tp[2] #y-axis
        
        sat_mask = np.zeros((self.shape[1:]), dtype='bool')
        sat_mask[tp1,tp2] = True

        aper_new2 = ndimage.binary_dilation(sat_mask, iterations=2) 

        inBackgroundAperture3 = inBackgroundAperture2 * ~aper_new2
        
        #Get indices of where array is true in this background mask
        bkg = np.where(inBackgroundAperture3==True)
        bkg2 = [bkg[0],bkg[1]]

        #For each pixel above get the median value of its time series - but only good quality data
        medval = []
        
        for a in range(self.shape[1]):
            custom_mask = np.zeros((self.shape[1:]), dtype='bool')
            custom_mask[bkg2[0][a],bkg2[1][a]] = True
            sap_lc = self.tpf[self.quality == 0].to_lightcurve(aperture_mask=custom_mask)
            medval.append(np.nanmedian(sap_lc.flux.value))
             

        #Now sort the data and get indicies
        sorted_index = np.argsort(medval)

        #Grab the 3rd lowest value
        vals2 = sorted_index[2]
        puse = medval[vals2]

        #The bgBias value is only important if it is less than zero
        #Need to do this check and let user know.
        if  puse >= 0:
            bgBias = 0
            print("The background bias estimate is greater than or equal to zero - no adjustment is required.")
            return bgBias
        else:
            bgBias = np.abs(puse)
         
            #Get the number of pixels in the SPOC optimal aperture
            nPix = np.sum(self.pipeline_mask)
        
            #Get the CROWDSAP function from the TPF
            CROWDSAP = self.tpf.hdu[1].header['CROWDSAP']
        
            #Get the FLFRCSAP function from the TPF
            FLFRCSAP = self.tpf.hdu[1].header['FLFRCSAP']

            #Get the median flux from the input PDCSAP_FLUX light curve
            medFluxPdc = np.nanmedian(self.lc[self.lc.quality == 0].flux.value)
        
            #Then calculate the pdcCorrection
            pdcCorrection = bgBias*nPix*CROWDSAP/FLFRCSAP

            #Correct the flux and make a new PDCSAP_FLUX light curve
            flux_corr = self.lc.flux.value + pdcCorrection
            corrected_lc = lk.LightCurve(time=self.lc.time.value, flux=flux_corr, flux_err=self.lc.flux_err.value)

            #plot this up
            ax = self.lc.scatter(label='PDCSAP')
            corrected_lc.scatter(ax=ax, color='red', label='PDCSAP CORR')

            #Calculate the transitBias
            transitBias = (pdcCorrection/medFluxPdc)*100

            # bgBias, pdcCorrection, transitBias, corrected_lc

            return  bgBias, pdcCorrection, transitBias, corrected_lc

    def diagnose(self):
        return None
