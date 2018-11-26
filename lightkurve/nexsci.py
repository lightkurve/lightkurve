import pandas as pd
import numpy as np
import astropy.units as u
import logging
import os
import datetime
from . import PACKAGEDIR

log = logging.getLogger(__name__)

__all__ = ['get_nexsci_data', 'create_planet_mask', 'find_planet_mask']

class OnlineRetrievalFailure(Exception):
    pass

def _fill_data(df):
    ''' Takes the NExSci dataframe and fills in any missing data with values.
    '''

    # Fill missing EqT
    nan = ~np.isfinite(df.pl_eqt)
    sep = np.asarray(df.pl_orbsmax)*u.AU
    rstar = (np.asarray(df.st_rad)*u.solRad).to(u.AU)
    temp = np.asarray(df.st_teff)*u.K
    df.loc[nan, ['pl_eqt']] = (temp[nan]*np.sqrt(rstar[nan]/(2*sep[nan])))

    # Fill in missing trandep
    nan = ~np.isfinite(df.pl_trandep)
    trandep = (np.asarray(df.pl_radj * u.jupiterRad.to(u.solRad))/np.asarray(df.st_rad))**2
    df.loc[nan, ['pl_trandep']] = trandep[nan]

    df['pl_density'] = (np.asarray(df.pl_bmassj)*u.jupiterMass.to(u.g))/((4/3) *
                        np.pi * (np.asarray(df.pl_radj)*u.jupiterRad.to(u.cm))**3)

    df['pl_rade'] = df.pl_radj * u.jupiterRad.to(u.earthRad)
    df['pl_bmasse'] = df.pl_bmassj * u.jupiterMass.to(u.earthMass)

    return df


def _retrieve_online_data():
    ''' Obtain a dataframe from NExScI of all planet data and store it in the
    package data directory.
    '''
    NEXSCI_API = 'http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
    try:
        planets = pd.read_csv(NEXSCI_API + '?table=planets&select=pl_hostname,pl_letter,'
                'pl_disc,ra,dec,pl_trandep,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,'
                'pl_tranflag,pl_trandur,pl_pnum,pl_k2flag,pl_kepflag,pl_facility,'
                'pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_orblper,st_mass,st_masserr1,'
                'st_masserr2,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,'
                'st_tefferr2,st_optmag,st_j,st_h', comment='#')
        composite = pd.read_csv(NEXSCI_API + '?table=compositepars&select=fpl_hostname,'
                    'fpl_letter,fpl_smax,fpl_smaxerr1,fpl_smaxerr2,fpl_radj,fpl_radjerr1,'
                    'fpl_radjerr2,fpl_bmassj,fpl_bmassjerr1,fpl_bmassjerr2,fpl_eqt,'
                    'fpl_orbper,fpl_orbpererr1,fpl_orbpererr2,fpl_eccen,'
                    'fpl_eccenerr1,fpl_eccenerr2, ', comment='#')
        composite.columns = ['pl_hostname', 'pl_letter', 'pl_orbsmax', 'pl_orbsmaxerr1',
                             'pl_orbsmaxerr2', 'pl_radj', 'pl_radjerr1', 'pl_radjerr2',
                             'pl_bmassj', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_eqt',
                             'pl_orbper', 'pl_orbpererr1', 'pl_orbpererr2', 'pl_eccen',
                             'pl_eccenerr1', 'pl_eccenerr2']
    except:
        raise OnlineRetrievalFailure("Couldn't obtain data from NExScI. Do you have an internet connection?")
    df = pd.merge(left=planets, right=composite, how='left', left_on=['pl_hostname', 'pl_letter'],
         right_on=['pl_hostname', 'pl_letter'])
    df = _fill_data(df)
    df[df.pl_tranflag == 1].to_csv("{}/data/planets.csv".format(PACKAGEDIR), index=False)


def _check_data_is_fresh():
    '''Checks whether there is a stored dataframe of planet data that is recent.
    If not, will attempt to redownload.
    '''
    fname = "{}/data/planets.csv".format(PACKAGEDIR)
    if len(fname) == 0:
        try:
            log.info('Retrieving data from NExScI')
            _retrieve_online_data()
        except OnlineRetrievalFailure:
            log.warning("Couldn't obtain data from NExScI. Do you have an internet connection?")
        fname = "{}/data/planets.csv".format(PACKAGEDIR)
    st = os.stat(fname)
    mtime = st.st_mtime
    # If database is out of date, get it again.
    if (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime) > datetime.timedelta(days=7)):
        log.warning('NExScI Database out of date. Redownloading...')
        try:
            log.info('Retrieving data from NExScI')
            _retrieve_online_data()
        except OnlineRetrievalFailure:
            log.warning("Couldn't obtain data from NExScI. Do you have an internet connection?")


def get_nexsci_data():
    '''Read and return the pandas dataframe of all exoplanet data
    '''
    _check_data_is_fresh()
    return pd.read_csv("{}/data/planets.csv".format(PACKAGEDIR))


def create_planet_mask(time_array, period, t0, duration):
    '''Returns a boolean mask for a given time. True where planets are not transiting.

    Parameters
    ----------
    time : np.ndarray of JD
        Array of time values at which to compute the mask. Note: Make sure that time is
        in JULIAN DAY.
    period : float
        Period of planet
    t0 : float
        Transit mid point
    duration : float
        Transit duration

    Returns
    -------
    mask : boolean array
        Array where True indicates there is no planet transiting. False indicates a planet is transiting.
    '''
    ph = (time_array - t0)/period % 1
    mask = (ph > ((duration * 1.5)/period)/2) & (ph < 1 - ((duration * 1.5)/period)/2)
    return mask


def find_planet_mask(name, time_array, planets='all'):
    '''Returns a boolean mask for a given time. True where planets are not transiting.

    Parameters
    ----------
    name: str
        Name of planet host as given by NexSCI
    time_array : np.ndarray of JD
        Array of time values at which to compute the mask. Note: Make sure that time is
        in JULIAN DAY.
    planets : str or list of indexes
        Array of which planets to compute the mask for. e.g. [0, 1] would calculate
        the mask for the first two planets in the system. Default is 'all'.

    Returns
    -------
    mask : boolean array
        Array where True indicates there is no planet transiting. False indicates a planet is transiting.
    '''

    df = get_nexsci_data()
    df = df[df.pl_hostname == name].reset_index(drop=True)
    df = df[['pl_orbper', 'pl_tranmid', 'pl_trandur']]
    if len(df) == 0:
        raise ValueError('No such planet as {}'.format(name))
    if isinstance(planets, str):
        if planets == 'all':
            planets = np.arange(0, len(df))
    if not isinstance(planets, (list, np.ndarray)):
        raise ValueError('Planets must be an iterable (list or numpy array)')
    else:
        if planets[-1] > len(df):
            raise ValueError('Only {} planet(s) in the {} system.'.format(len(df), name))
    if (~np.isfinite(np.asarray(df))).any():
        raise ValueError('Some ephemeris values are NaNs. Cannot compute mas')

    mask = np.ones(len(time_array), dtype=bool)
    for idx in planets:
        period, t0, duration = df.iloc[idx]
        mask &= create_planet_mask(time_array, period, t0, duration)
    return mask
