from abc import abstractmethod, ABCMeta
import logging

log = logging.getLogger(__name__)

__all__ = ['Collection', 'LightCurveCollection', 'TargetPixelFileCollection']

class Collection:
    """Abstract Base Class for LightCurveCollections, TPFCollections

    Attributes
    ----------
    data: array
        List of data objects.
    k_id: dictionary
        Mapping target to index in self.data
    """
    __metaclass__ = ABCMeta #Needs to be set for Python 2.x to work properly
    def __init__(self, data):
        self.data = data
        self.k_id = self._update_target_map()

    def _update_target_map(self):
        """
        Assigns the targetid to indexes in self.data

        Parameters
        ----------
        None

        Returns
        -------
        result: Dictionary
            With keys of targetid (int) and
            values of indexes (int) in the data array.
        """
        result = {}
        for idx,obj in enumerate(self.data):
            try:
                result[obj.targetid] = idx
            except AttributeError:
                print("Object "+ str(idx) + " has no targetid")
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Called when accessing an element in the collection
        by index or target id

        Parameters
        ----------
        index: int
            This is either the index of the element in the collection,
            or the target id of the object

        Returns
        -------
        Lightcurve or TargetPixelFile object

        Raises KeyError when given trying to index with an invalid array position
        or targetid
        """
        try:
            if index > len(self.data):
                return self.data[self.k_id[index]]
            else:
                return self.data[index]
        except:
            raise KeyError("Invalid key")

    def __setitem__(self, key, value):
        """Implicitly called during assignment

        Parameters
        ----------
        key: int
            Array slice or targetid
        value: lightcurve or target pixel file object
        """
        if key > len(self.data):
            #indexed by targetid
            self.data[self.k_id[key]] = value
            self.k_id = self._update_target_map()
        else:
            #indexed by array
            self.data[key] = value
            self.k_id = self._update_target_map()

    def append(self, obj):
        """Adds a new object to the collection.

        Parameters
        ----------
        obj: LightCurve or TargetPixelFile object

        Returns
        -------
        None
        """
        self.data.append(obj)
        if obj.targetid and obj.targetid in self.k_id:
            raise AttributeError("Cannot add multiple objects with same targetid")
        try:
            self.k_id[obj.targetid] = len(self.data)-1
        except AttributeError:
            print("Object has no targetid")

    def __repr__(self):
        """Used in printing

        Used to print out all the items in lcc

        Returns:
        result: str
            String containing the resulting lightcurves.
        """
        result = ""
        for obj in self.data:
            result += obj.__repr__() + " "
            result += "\n"
        return result

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        pass

class LightCurveCollection(Collection):
    """Represents a set of LightCurves

    Attributes
    ----------
    data: array
        List of lightcurve objects.
    k_id: dictionary
        Mapping targetid to index in self.data
    """
    def __init__(self, lightcurves):
        super(LightCurveCollection, self).__init__(lightcurves)

    def plot(self, ax=None, **kwargs):
        """Plots a collection of light curve.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.

        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if ax is None:
            _, ax = plt.subplots()
        for lightcurve in self.data:
           lightcurve.plot(ax=ax, label=lightcurve.targetid)
        return ax

    def pca(self):
        '''Creates the Principle Components of a collection of LightCurves
        '''
        raise NotImplementedError('Should be able to run a PCA on a collection.')

class TargetPixelFileCollection(Collection):
    """Represents a set of Target Pixel Files

    Attributes
    ----------
    data: array
        List of TPF objects.
    k_id: dictionary
        Mapping targetid to index in self.data
    """
    def __init__(self, tpfs):
        super(TargetPixelFileCollection, self).__init__(tpfs)

    def plot(self, ax=None, **kwargs):
        """Plots a collection of TPF objects in space.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.

        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
        """
        raise NotImplementedError('Plotting TPFs has not been implemented')
