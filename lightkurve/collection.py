from abc import abstractmethod, ABCMeta

__all__ = ['Collection']

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
        self.k_id = self.assign_hash_values()

    def assign_hash_values(self):
        """
        Assigns the targetid to indexes in self.data

        Parameters
        ----------
        None

        Returns
        -------
        result: Dictionary
            With keys of keplerid (int) and
            values of indexes (int) in the data array.
        """
        result = {}
        for idx,obj in enumerate(self.data):
            try:
                result[obj.keplerid] = idx
            except AttributeError:
                print("Object "+ str(idx) + " has no keplerid")
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
        """
        try:
            if index > len(self.data):

                return self.data[self.k_id[index]]
            else:
                return self.data[index]
        except:
            raise KeyError("The collection does not contain an object with target ID {}.".format(self.k_id[index]))

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
        try:
            self.k_id[obj.keplerid] = len(self.data)-1
        except AttributeError:
            print("Object has no keplerid")

    def __repr__(self):
        """Used in printing

        Used to print out all the items in lcc

        Returns:
        result: str
            String containing the resulting lightcurves.
        """
        result = ""
        for obj in self.data:
            results += obj.__repr__() + " "
            result += "\n"
        return result

    @abstractmethod
    def plot(self, ax=None, **kwargs):
        pass
