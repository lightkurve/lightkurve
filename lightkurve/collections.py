"""Defines collections of data products."""
from abc import ABCMeta
import logging
import matplotlib.pyplot as plt
import numpy as np

from . import MPLSTYLE

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
        for obj in self.data:
            result += obj.__repr__() + " "
            result += "\n"
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

        if self[0].mission == 'Kepler':
            # If mission is kepler, find offsets by sky group and correct them
            # FLFRCSAP and CROWDSAP have been applied, but in some cases there is
            # Residual multiplicative flux loss.

            groups = np.asarray([lcf.quarter for lcf in self]) % 4
            groups[np.asarray([lcf.quarter for lcf in self]) == 0] = 4
            corr = np.zeros(5)
            for skygroup in np.arange(5):
                idxs = np.where(groups == skygroup)[0]
                if not hasattr(idxs, '__iter__'):
                    idxs = [idxs]
                t1, f1 = [np.nanmedian(self[idx].time) for idx in idxs], [np.nanmedian(self[idx].flux) for idx in idxs]
                if len(t1) == 1:
                    corr[skygroup] = f1[0]
                else:
                    corr[skygroup] = np.polyfit(t1, f1, 1)[1]
            corr /= np.mean(corr)
            lcs = [corrector_func(lc/corr[groups[idx]]) for idx, lc in enumerate(self)]
#            lcs = [corrector_func(lc) for idx, lc in enumerate(self)]

        else:
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
            for idx, targetid in enumerate(np.unique(targetids)):
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


    def plot(self, offset=0.1, **kwargs):
        try:
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
