"""Defines collections of data products."""
from abc import ABCMeta
import logging

log = logging.getLogger(__name__)

__all__ = ['LightCurveCollection', 'TargetPixelFileCollection']


class Collection:
    """Abstract Base Class for LightCurveCollection and TargetPixelFileCollection.

    Attributes
    ----------
    data: array-like
        List of data objects.
    """
    __metaclass__ = ABCMeta  # Needs to be set for Python 2.x to work properly

    def __init__(self, data):
        self.data = data
        # At any given time, `self._targetid_map` maps targetid's onto the
        # indexes into self.data.
        self._targetid_map = self._create_targetid_map()

    def _create_targetid_map(self):
        """Returns a dictionary that maps targetid's onto `self.data` indexes.

        Returns
        -------
        result: Dictionary
            The keys are targetid's and the values are indexes (int) into
            the `self.data` array.
        """
        targetid_map = {}
        for idx, obj in enumerate(self.data):
            try:
                targetid_map[obj.targetid] = idx
            except AttributeError:
                pass
        return targetid_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index_or_targetid):
        """Access an element by index or targetid.

        Parameters
        ----------
        index_or_targetid: int or str
            This is either the index of the element in the collection,
            or the targetid of the object.

        Returns
        -------
        data : object
            Typically a LightCurve or TargetPixelFile object.

        Raises
        ------
        KeyError
            If the index or targetid does not exist.
        """
        try:
            return self.data[self._targetid_map[index_or_targetid]]
        except KeyError:
            try:
                return self.data[index_or_targetid]
            except IndexError:
                raise KeyError("{} is not a valid index or targetid".format(index_or_targetid))

    def __setitem__(self, index_or_targetid, obj):
        """Set an item in the collection.

        Parameters
        ----------
        index_or_targetid : int or str
            Index or targetid.
        obj : object
            Typically a LightCurve or TargetPixelFile object
        """
        if index_or_targetid in self._targetid_map.keys():
            self.data[self._targetid_map[index_or_targetid]] = obj
        else:
            self.data[index_or_targetid] = obj
        self._targetid_map = self._create_targetid_map()

    def append(self, obj):
        """Appends a new object to the collection.

        Parameters
        ----------
        obj : object
            Typically a LightCurve or TargetPixelFile object
        """
        self.data.append(obj)
        self._targetid_map = self._create_targetid_map()

    def __repr__(self):
        """Used in printing

        Used to print out all the items in lcc

        Returns:
        result: str
            String containing the resulting lightcurves.
        """
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


class TargetPixelFileCollection(Collection):
    """Class to hold a collection of TargetPixelFile objects.

    Parameters
    ----------
    tpfs : array-like
        List of TargetPixelFile objects.
    """
    def __init__(self, tpfs):
        super(TargetPixelFileCollection, self).__init__(tpfs)
