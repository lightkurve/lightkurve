"""Defines collections of data products."""
from abc import ABCMeta
import logging

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

    def plot(self, ax=None, **kwargs):
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
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
        for lc in self.data:
            lc.plot(ax=ax, label=lc.targetid, **kwargs)
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

class TargetPixelFileCollection(Collection):
    """Class to hold a collection of TargetPixelFile objects.

    Parameters
    ----------
    tpfs : array-like
        List of TargetPixelFile objects.
    """
    def __init__(self, tpfs):
        super(TargetPixelFileCollection, self).__init__(tpfs)

    def plot_all(self, ax=None):
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
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(len(self.data), 1,
                                 figsize=(7, (7*len(self.data))))

        if len(self.data) == 1:
            self.data[0].plot(ax=ax)
        else:
            for i, tpf in enumerate(self.data):
                tpf.plot(ax=ax[i])
        return ax
