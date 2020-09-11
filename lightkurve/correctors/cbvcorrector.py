"""Defines Corrector classes that utilize Kepler/K2/TESS Cotrending Basis Vectors.
"""
import logging
import copy
import requests
import urllib.request

from astropy.io import fits as pyfits
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.units import Quantity

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from .designmatrix import DesignMatrix, DesignMatrixCollection
from .. import MPLSTYLE
from ..lightcurve import LightCurve
from ..utils import channel_to_module_output, validate_method

log = logging.getLogger(__name__)

__all__ = ['CotrendingBasisVectors', 'KeplerCotrendingBasisVectors',
           'TessCotrendingBasisVectors', 'download_kepler_cbvs', 'download_tess_cbvs']

#*******************************************************************************
# Cotrending Basis Vectors Classes and Functions 

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
            timeArray = self.time.copy().value
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

        if not isinstance(lc, LightCurve):
            raise Exception('<lc> must be a LightCurve class')

        if hasattr(lc, 'cadenceno'):

            # Make a deepcopy so we do not just return a modified original
            cbvs = copy.deepcopy(self)

            trim_indices = np.nonzero(np.logical_not(
                np.in1d(cbvs.cadenceno, lc.cadenceno)))[0]
            cbvs.remove_rows(trim_indices)

            # NaN any cadences in light curve and not in CBVs
            # This requires us to add rows to the CBV table
            lc_nan_mask = np.logical_not(np.in1d(lc.cadenceno, cbvs.cadenceno))
            lc_nan_indices = np.nonzero(lc_nan_mask)[0]
            # Sadly, there is no TimesSeries.add_rows (plural), so we have to
            # add each row in a for-loop
            if len(lc_nan_indices) > 0:
                for idx in lc_nan_indices:
                    dict_to_add = {}
                    dict_to_add['time'] = lc.time[idx]
                    dict_to_add['CADENCENO'] = lc.cadenceno[idx].value
                    dict_to_add['GAP'] = True
                    for cbvIdx in cbvs.cbv_indices:
                        dict_to_add['VECTOR_{}'.format(cbvIdx)] = np.nan
                
                    cbvs.add_row(dict_to_add)

            # Now sort the CBVs by cadenceno
            cbvs.sort('CADENCENO')

        else:
            raise Exception('align requires cadence numbers for the ' + \
                    'light curve. NO SYNCHRONIZATION OCCURED')

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
        
        # Create the new cbv object with no basis vectors, yet...
        cbvNewTime = lc.time.copy()
        # Gaps are all false
        gaps = np.full(len(lc.time), False)
        dataTbl = Table([lc.cadenceno, gaps], names=('CADENCENO', 'GAP'))

        # We are PCHIP interpolating each CBV independently.
        for idx in self.cbv_indices:
            fInterp = PchipInterpolator(self.time.value,
                    self['VECTOR_{}'.format(idx)], extrapolate=extrapolate)
            dataTbl['VECTOR_{}'.format(idx)] = fInterp(lc.time.value)

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
            dataTbl = Table.read(hdu[extName], format="fits")
            dataTbl.meta.update(hdu[0].header)
            dataTbl.meta.update(hdu[extName].header)
  
            # TimeSeries-based objects require a dedicated time column
            # Replace NaNs with default time '2000-01-01', otherwise,
            # astropy.time.Time complains
            nanHere = np.nonzero(np.isnan(dataTbl['TIME_MJD'].data))[0]
            timeData = dataTbl['TIME_MJD'].data
            timeData[nanHere] = Time(['2000-01-01'], scale='utc').mjd
            cbvTime = Time(timeData, format='mjd')
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

        TESS CBVs are in seperate FITS files for each camera.CCD, so camera.CCD
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
            dataTbl = Table.read(hdu[extName], format="fits")
            dataTbl.meta.update(hdu[0].header)
            dataTbl.meta.update(hdu[extName].header)
  
            # TimeSeries-based objects require a dedicated time column
            # Replace NaNs with default time '2000-01-01', otherwise,
            # astropy.time.Time complains
            nanHere = np.nonzero(np.isnan(dataTbl['TIME'].data))[0]
            timeData = dataTbl['TIME'].data
            timeData[nanHere] = Time(['2000-01-01'], scale='utc').mjd
            cbvTime = Time(timeData, format='btjd')
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

def download_kepler_cbvs(mission=None, quarter=None, campaign=None,
        channel=None, module=None, output=None):
    """Searches the public data archive at MAST <https://archive.stsci.edu>
    for Kepler or K2 cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    mission, quarter/campaign and channel or module/output, etc...
    and then extracts the requested basis vectors and returns a
    KeplerCotrendingBasisVectors object

    For Kepler/K2, the FITS files contain all channels in a single file per
    quarter/campaign.

    For Kepler this extracts the DR25 CBVs.

    Parameters
    ----------
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

        >>> cbvs = download_kepler_cbvs(mission='Kepler', quarter=8, module=16, output=4) # doctest: +SKIP

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
        raise Exception('Unknown mission type')

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

    if (mission == 'Kepler'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/kepler/cbv/"
    elif (mission == 'K2'):
        cbvBaseUrl = "http://archive.stsci.edu/missions/k2/cbv/"

    try:     
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

        kepler_cbv_url = cbvBaseUrl + cbv_file
        hdu = pyfits.open(kepler_cbv_url)

        return KeplerCotrendingBasisVectors.from_hdu(hdu=hdu, module=module, output=output)

    except:
        raise Exception('CBVS were not found')


def download_tess_cbvs(sector=None, camera=None,
        ccd=None, cbv_type='SingleScale', band=None):
    """Searches the `public data archive at MAST <https://archive.stsci.edu>`
    for TESS cotrending basis vectors.

    This function fetches the Cotrending Basis Vectors FITS HDU for the desired
    cotrending basis vectors.

    For TESS, each CCD CBVs are stored in a seperate FITS files.

    For now, this function will only download 2-minute cadence CBVs. Once other
    cadence CBVs become available this function will be updated to support
    their downloads.

    Parameters
    ----------
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
    This example will read in the CBVs for TESS Sector 10 Camera.CCD 2.4
    Multi-Scale band 2

        >>> cbvs = download_tess_cbvs(sector=10, camera=2, ccd=4, # doctest: +SKIP
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
        
    curlBaseUrl = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_'
    curlEndUrl = '_cbv.sh'
    curlUrl = curlBaseUrl + str(sector) + curlEndUrl

    # This is the string to search for in the curl script file
    # Pad the sector number with a first '0' if less than 10
    # TODO: figure out a way to pad an integer number with forward zeros
    # without needing a conditional
    sector = int(sector)
    if (sector < 10):
        curlSearchString = 's000' + str(sector) + '-' + str(camera) + '-' + str(ccd) + '-'
    elif (sector >= 10 and sector < 100):
        curlSearchString = 's00' + str(sector) + '-' + str(camera) + '-' + str(ccd) + '-'
    elif (sector >= 100 and sector < 1000):
        curlSearchString = 's0' + str(sector) + '-' + str(camera) + '-' + str(ccd) + '-'
    elif (sector > 999):
        # TESS will be truly blessed if it gets to more than 999 sectors!
        raise Exception('Only up to 999 Sectors is currently supported')
    else:
        raise Exception('Error parsing sector string when getting TESS CBV FITS files')

    try: 

        # Read in the relevent curl script file and find the line for the CBV 
        # data we are looking for
        data = urllib.request.urlopen(curlUrl)
        foundIndex = None
        for line in data:
            strLine = str(line)
            if curlSearchString in strLine: 
                foundIndex = strLine.index(curlSearchString)
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
