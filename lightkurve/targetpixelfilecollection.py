from __future__ import division, print_function

import copy
from tqdm import tqdm
import os
import datetime
import logging
from collections import Sequence

import oktopus
import numpy as np
from scipy import signal
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib as mpl

from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time


__all__ = ['LightCurve', 'KeplerLightCurve', 'TessLightCurve',
           'iterative_box_period_search']

log = logging.getLogger(__name__)

class TargetPixelFileCollection(Sequence):
    """
    Collects multiple TPF objects together with helpful functions.
    """

    def __init__(self, lcs):
        tpfs = np.asarray(lcs)
        self.data = {}
        for TPF in tpfs:
            if TPF.keplerid:
                self.data[TPF.keplerid] = TPF

    def __len__(self):
        return len(self.data)

    def _ids(self):
        '''
        Returns the kepler_ids of all the lightcurves as a dict_keys obj.
        '''
        return self.data.keys()

    def __getitem__(self, kep_id):
        '''
        Returns the lightcurve associated with the kepler_id. 
        '''
        try: 
            return self.data[kep_id]
        except:
            raise ValueError('No TPF for ' + kep_id)

    def append(self, tpf):
        try:
            self.data[tpf.keplerid] = tpf
        except:
            log.warning("Input is not a TPF")

    def __repr__(self):
        result = ""
        for TPF in self.data.values():
            result += TPF.__repr__() + "\n"
        return result

    def plot(self):
        '''
        Plot a collection of LightCurves. Random colors are assigned to each plot.
        '''
        for i, k_id in enumerate(self.data):
            if i == 0:
                #TODO: what if longest axis is not the first item?
                axis = self.data[k_id].plot(label=k_id)
            else:
                self.data[k_id].plot(ax=axis, label=k_id)


    def pca(self):
        '''Creates the Principle Components of a collection of LightCurves
        '''
        raise NotImplementedError('Should be able to run a PCA on a collection.')

