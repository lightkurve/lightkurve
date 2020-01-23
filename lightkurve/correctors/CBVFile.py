"""
Defines the CBVFile classes.

These are used to retrieve Cotrending Basis Vectors (CBVs) from the archive.

The CBVs are then to be incoroporated into a CotrendingBasisVectors object before use to correct light curves.

"""

import logging
import requests
import numpy as np
from astropy.io import fits as pyfits
from bs4 import BeautifulSoup
import urllib.request

from ..utils import channel_to_module_output
from .CBV import KeplerCotrendingBasisVectors, TessCotrendingBasisVectors

__all__ = ['CBVFile', 'KeplerCBVFile', 'TessCBVFile']

log = logging.getLogger(__name__)

class CBVFile(object):
    """ Generic class for storing the data from CBV files from the archive

    """

    #***
    # Some constants
    @property
    def nCBVsDefault():
        """ Default number of CBVs """
        # For Kepler/K2/TESS it's always been 16 CBVs
        return 16
    nCBVsDefault = staticmethod(nCBVsDefault)

    #***
    def __init__(self, path, **kwargs):
        """ Contructor: Fetches a CBV FITS file from the archive 

        Parameters
        ----------
        path : str or `astropy.io.fits.HDUList` object
            Local path or remote url of a lightcurve FITS file.
            Also accepts a FITS file object already opened using AstroPy.
        kwargs : dict
            Keyword arguments to be passed to astropy.io.fits.open.
        """
        if isinstance(path, pyfits.HDUList):
            self.path = None
            self.hdu = path
        else:
            self.path = path
            self.hdu = pyfits.open(self.path, **kwargs)

    def header(self, ext=0):
        """Header of the object at extension `ext`"""
        return self.hdu[ext].header

    def get_keyword(self, keyword, ext=0, default=None):
        """Returns a header keyword value.

        If the keyword is Undefined or does not exist,
        then return ``default`` instead.
        """
        try:
            kw = self.hdu[ext].header[keyword]
        except KeyError:
            return default
        if isinstance(kw, Undefined):
            return default
        return kw

    def get_cbvs (self, cbvType='singleScale', cbvIndices='ALL'):

        pass


        
#*************************************************************************************************************
class KeplerCBVFile(CBVFile):
    """ Subclass for Kepler CBVs
 
         Only the singleScale CBVs are archived for Kepler
 
    """
 
    def __init__ (self, path, mission=None):
        """ Constructor:

            There is no "MISSION" keyword in the CBV FITS files so that must be explicitely given
        """

        super(KeplerCBVFile, self).__init__(path)
        
        assert not mission is None, log.error('"mission" must be passed for Kepler/K2 CBV Fits files')
        self.mission = mission

        pass

    def get_cbvs(self, channel=None, module=None, output=None, cbvIndices='ALL'):
        """ Returns the requested CBVs as a CotrendingBasisVectors object

            Parameters
            ----------
                channel     : [int] If passed then module and output must NOT be
                module      : [int] If passed then channel must Not be
                output      : [int] If passed then channel must Not be
                cbvIndices  : [int array] List of CBVs to get (does not need to be in sequential order, and can skip indices)
        """

        # CBV FITS files use module/output, not channel
        # So if channel is passed, convert to module/output
        if (isinstance(channel, int)):
            assert  module is None, 'module must NOT be passed if channel is passed'
            assert  output is None, 'output must NOT be passed if channel is passed'
            module, output = channel_to_module_output(channel)
            channel = None
        else:
            assert  not module is None, 'module must be passed'
            assert  not output is None, 'output must be passed'


        cbvs = KeplerCotrendingBasisVectors(self.mission, self.hdu, module, output, cbvIndices=cbvIndices)
 
        return cbvs


    @staticmethod
    def get_kepler_cbv_url(mission, quarter, campaign):
        """ STATIC method to obtain a path to a Kepler/K2 CBV FITS file

            For Kepler extracts the DR25 CBVs

            Gets the html page and finds all references to 'a' tag
            keeps the ones for which 'href' ends with 'fits'
            this might slow things down in case the user wants to fit 1e3 stars
        """
 
        if (mission == 'Kepler'): 
            cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
        elif (mission == 'K2'):
            cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"
 
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
        return cbvBaseUrl + cbv_file


#*************************************************************************************************************
class TessCBVFile(CBVFile):
    """ Subclass for TESS CBVs
 
 
    """
 
    def __init__ (self, path):
        """ Constructor:

            There is no "MISSION" keyword in the CBV FITS files so that must be explicitely given
            leyword MISSION is not technically in the TESS CBV FITS files but 'TELESCOP' is in there whcih gives what we
            need.
        """

        super(TessCBVFile, self).__init__(path)
        
        # Check that this is a TESS CBV FITS file
        self.mission = self.hdu['Primary'].header['TELESCOP']
        assert self.mission == 'TESS', log.error('This does not appear to be a TESS FITS HDU')
        
        pass

    def get_cbvs(self, cbvType='SingleScale', band=None, cbvIndices='ALL'):
        """ Returns the requested TESS CBVs as a CotrendingBasisVectors object

            Input:
                cbvType     -- [str ('SingleScale', 'MultiScale', 'Spike')
                band        -- [int] MultiScale band number (invalid for other CBV types)
                cbvIndices  -- [int arry] List of CBVs extracted for FITS file, {'ALL' => extract all}
                                    (does not need to be in sequential order, and can skip indices)
        """

        cbvs = TessCotrendingBasisVectors(self.hdu, cbvType, band, cbvIndices=cbvIndices)
 
        return cbvs


    @staticmethod
    def get_tess_cbv_url(sector, camera, CCD):
        """ STATIC method to obtain a path to a TESS CBV FITS file

            The easiest way to obtain a link to the CBV file for a TESS Sector and camera.CCD is

            1. Download the bulk download curl script (with a predictable url) for the desired sector and search it for the camera.CCD I need
            2. Download the CBV FITS file based on the link in the curl script

            The bulk download curl links have urls such as:

            https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_17_cbv.sh

            Then the individual CBV files foudn in the curl file have urls such as:

            https://archive.stsci.edu/missions/tess/ffi/s0017/2019/279/1-1/tess2019279210107-s0017-1-1-0161-s_cbv.fits

            Returns a url string for the desired CBV FITS file
    
        """
        curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
        curlEndUrl = '_cbv.sh'
        curlUrl = curlBaseUrl + str(sector) + curlEndUrl

        # This is the string to search for in the curl script file
        curlSearchString = 's00' + str(sector) + '-' + str(camera) + '-' + str(CCD) + '-' 

        # 1. Read in the relevent curle script file and fine the line for the CBV data we are looking for
        data = urllib.request.urlopen(curlUrl)
        foundIndex = None
        for line in data:
            strLine = str(line)
            try:
                foundIndex = strLine.index(curlSearchString) # str.index will error when not found
                break
            except:
                pass # continue searching
        if (foundIndex is None):
            log.error('CBV FITS file not found')

        # extract url from strLine
        htmlStartIndex = strLine.find('https:')
        htmlEndIndex = strLine.rfind('fits')
        cbvUrl = strLine[htmlStartIndex:htmlEndIndex+4] # Add 4 for length of 'fits' string

        return cbvUrl

