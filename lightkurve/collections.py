"""Defines collections of data products."""
from abc import ABCMeta
import logging
import matplotlib.pyplot as plt
import numpy as np

from . import MPLSTYLE
from .utils import LightkurveWarning

from .lightcurvefile import LightCurveFile

import warnings


log = logging.getLogger(__name__)

__all__ = ['LightCurveCollection', 'TargetPixelFileCollection']


class Collection(object):
    """Abstract Base Class for LightCurveCollection and TargetPixelFileCollection.

    Attributes
    ----------
    data: array-like
        List of data objects.
    """
#    __metaclass__ = ABCMeta  # Needs to be set for Python 2.x to work properly

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, obj):
        self.data[index] = obj

    def append(self, obj):
        """Appends a new object to the collection.

        Parameters
        ----------
        obj : object
            Typically a LightCurve or TargetPixelFile object
        """
        self.data.append(obj)

    def __repr__(self):
        result = "{} of {} objects:\n".format(self.__class__.__name__, len(self.data))
        if (isinstance(self[0], LightCurveFile)):
            targetids = np.asarray([lcf.SAP_FLUX.label for lcf in self])
        else:
            targetids = np.asarray([lcf.label for lcf in self])

        try:
            unique_targetids = np.sort(np.unique(targetids))
        except TypeError:
            unique_targetids = [None]

        for idx, targetid in enumerate(unique_targetids):
            jdxs = np.where(targetids == targetid)[0]
            if not hasattr(jdxs, '__iter__'):
                jdxs = [jdxs]

            if hasattr(self[jdxs[0]], 'mission'):
                mission = self[jdxs[0]].mission
                if mission == 'Kepler':
                    subtype = 'Quarters'
                elif mission == 'K2':
                    subtype = 'Campaigns'
                elif mission == 'TESS':
                    subtype = 'Sectors'
                else:
                    subtype = None
            else:
                subtype = None
            objstr = str(type(self[0]))[8:-2].split('.')[-1]
            title = '\t{} ({} {}s) {}: '.format(targetid, len(jdxs), objstr, subtype)
            result += title
            if subtype is not None:
                result += ','.join(['{}'.format(getattr(self[jdx], subtype[:-1].lower())) for jdx in jdxs])
            else:
                result += ','.join(['{}'.format(i) for i in np.arange(len(jdxs))])
            result += '\n'
#        for obj in self.data:
#            result += obj.__repr__() + " "
#            result += "\n"
        return result


class LightCurveCollection(Collection):
    """Class to hold a collection of LightCurve objects.

    Attributes
    ----------
    lightcurves : array-like
        List of LightCurve objects.
    """
    def __init__(self, lightcurves):
        super(LightCurveCollection, self).__init__(lightcurves)


    def stitch(self, corrector_func=lambda x:x.normalize()):
        """ Stitch all light curves in the collection into a single lk.LightCurve

        Any function passed to `corrector_func` will be applied to each light curve
        before stitching. For example, passing "lambda x: x.normalize().flatten()"
        will normalize and flatten each light curve before stitching.

        Parameters
        ----------
        corrector_func : function taking in a lk.LightCurve and returning a lk.Lightcurve
            Corrector function that applies some correction to each light curve before
            stitching. Default is to normalize each light curve.

        Returns
        -------
        lc : lk.LightCurve
            Stitched Light Curve
        """
        if corrector_func is None:
            corrector_func = lambda x: x

        ## STOP IF THERE ARE MULTIPLE MISSIONS
        try:
            targets = np.unique([lc.label for lc in self])
        except TypeError:
            targets = [None]

        if len(targets) > 1:
            raise ValueError('This collection contains more than one target, '
                            'please reduce to a single target before calling stitch.')
        lcs = [corrector_func(lc) for lc in self]
        lc = lcs[0]
        [lc.append(lc1, inplace=True) for lc1 in lcs[1:]]
        return lc

    def plot(self, ax=None, offset=0.1, **kwargs):
        """Plots all lightcurves in the collection on a single plot.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.

        **kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            for kwarg in ['c', 'color', 'label', 'normalize']:
                if kwarg in kwargs:
                    kwargs.pop(kwarg)

            targetids = np.asarray([lcf.label for lcf in self])
            try:
                unique_targetids = np.sort(np.unique(targetids))
            except TypeError:
                unique_targetids = [None]
            for idx, targetid in enumerate(unique_targetids):
                jdxs = np.where(targetids == targetid)[0]
                if not hasattr(jdxs, '__iter__'):
                    jdxs = [jdxs]
                for jdx in jdxs:
                    if jdx == jdxs[0]:
                        (self[jdx].normalize() + idx*offset).plot(ax=ax, c='C{}'.format(idx), normalize=False, **kwargs)
                    else:
                        (self[jdx].normalize() + idx*offset).plot(ax=ax, c='C{}'.format(idx), normalize=False, label='', **kwargs)
        return ax


class LightCurveFileCollection(Collection):
    """Class to hold a collection of LightCurveFile objects.

    Parameters
    ----------
    lightcurvefiles : array-like
        List of KeplerLightCurveFile or TessLightCurveFile objects.
    """

    def __init__(self, lightcurvefiles):
        super(LightCurveFileCollection, self).__init__(lightcurvefiles)

    @property
    def PDCSAP_FLUX(self):
        return LightCurveCollection([lcf.PDCSAP_FLUX for lcf in self])

    @property
    def SAP_FLUX(self):
        return LightCurveCollection([lcf.SAP_FLUX for lcf in self])

    def stitch(self):
        """ Stitch all PDCSAP_FLUX extensions in the collection into a single lk.LightCurve

        Note: if SAP flux is required, use lcfs.SAP_FLUX.stitch() instead.
        """
        try:
            warnings.warn('Stitching a `LightCurveFileCollection` which contains both SAP and '
                         'PDCSAP_FLUX. Plotting PDCSAP_FLUX. You can remove this warning by '
                         'using `LightCurveFileCollection.PDCSAP_FLUX.stitch()`.',
                         LightkurveWarning)
            return self.PDCSAP_FLUX.stitch()
        except ValueError:
            return self.SAP_FLUX.stitch()

    def plot(self, offset=0.1, **kwargs):
        try:
            warnings.warn('Plotting a `LightCurveFileCollection` which contains both SAP and '
                         'PDCSAP_FLUX. Plotting PDCSAP_FLUX. You can remove this warning by '
                         'using `LightCurveFileCollection.PDCSAP_FLUX.plot()`.',
                         LightkurveWarning)
            ax = self.PDCSAP_FLUX.plot()
            return ax
        except ValueError:
            ax = self.SAP_FLUX.plot()
            return ax

class TargetPixelFileCollection(Collection):
    """Class to hold a collection of TargetPixelFile objects.

    Parameters
    ----------
    tpfs : array-like
        List of TargetPixelFile objects.
    """
    def __init__(self, tpfs):
        super(TargetPixelFileCollection, self).__init__(tpfs)

    def plot(self, ax=None):
        """Individually plots all TargetPixelFile objects in a single
        matplotlib axes object.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """

        if ax is None:
            _, ax = plt.subplots(len(self.data), 1,
                                 figsize=(7, (7*len(self.data))))

        if len(self.data) == 1:
            self.data[0].plot(ax=ax)
        else:
            for i, tpf in enumerate(self.data):
                tpf.plot(ax=ax[i])
        return ax
