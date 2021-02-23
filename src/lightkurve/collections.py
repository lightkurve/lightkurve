"""Defines collections of data products."""
import logging
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import vstack
from astropy.utils.decorators import deprecated

from . import MPLSTYLE
from .targetpixelfile import TargetPixelFile
from .utils import LightkurveDeprecationWarning


log = logging.getLogger(__name__)

__all__ = ["LightCurveCollection", "TargetPixelFileCollection"]


class Collection(object):
    """Base class for `LightCurveCollection` and `TargetPixelFileCollection`.

    A collection can be indexed by standard Python list syntax.
    Additionally, it can be indexed by a subset of `numpy.ndarray` syntax: boolean array indexing and integer array indexing.

    Attributes
    ----------
    data: array-like
        List of data objects.

    Examples
    --------
    Filter a collection by boolean array indexing.

        >>> lcc_filtered = lcc[(lcc.sector >= 13) & (lcc.sector <= 19)]  # doctest: +SKIP
        >>> lc22 = lcc[lcc.sector == 22][0]  # doctest: +SKIP

    Filter a collection by integer array indexing to get the object at index 0 and 2.

        >>>  lcc_filtered = lcc[0, 2]  # doctest: +SKIP
    """

    def __init__(self, data):
        if data is not None:
            # ensure we have our own container
            self.data = [item for item in data]
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index_or_mask):
        if isinstance(index_or_mask, (int, np.integer, slice)):
            return self.data[index_or_mask]
        elif all([isinstance(i, (bool, np.bool_)) for i in index_or_mask]):
            # case indexOrMask is bool array like, e.g., np.ndarray, collections.abc.Sequence, etc.

            # note: filter using nd.array is very slow
            #   np.array(self.data)[np.nonzero(indexOrMask)]
            # specifically, nd.array(self.data) is very slow, as it deep copies the data
            # so we create the filtered list on our own
            if len(index_or_mask) != len(self.data):
                raise IndexError(
                    f"boolean index did not match indexed array; dimension is {len(self.data)} "
                    f"but corresponding boolean dimension is {len(index_or_mask)}"
                )
            return type(self)([self.data[i] for i in np.nonzero(index_or_mask)[0]])
        elif all([isinstance(i, (int, np.integer)) for i in index_or_mask]):
            # case int array like, follow ndarray behavior
            return type(self)([self.data[i] for i in index_or_mask])
        else:
            raise IndexError(
                "only integers, slices (`:`) and integer or boolean arrays are valid indices"
            )

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
        if isinstance(self[0], TargetPixelFile):
            labels = np.asarray([tpf.targetid for tpf in self])
        else:
            labels = np.asarray([lc.meta.get("LABEL") for lc in self])

        try:
            unique_labels = np.sort(np.unique(labels))
        except TypeError:
            unique_labels = [None]

        for idx, targetid in enumerate(unique_labels):
            jdxs = np.where(labels == targetid)[0]
            if not hasattr(jdxs, "__iter__"):
                jdxs = [jdxs]

            if hasattr(self[jdxs[0]], "mission"):
                mission = self[jdxs[0]].mission
                if mission == "Kepler":
                    subtype = "Quarters"
                elif mission == "K2":
                    subtype = "Campaigns"
                elif mission == "TESS":
                    subtype = "Sectors"
                else:
                    subtype = None
            else:
                subtype = None
            objstr = str(type(self[0]))[8:-2].split(".")[-1]
            title = "\t{} ({} {}s) {}: ".format(targetid, len(jdxs), objstr, subtype)
            result += title
            if subtype is not None:
                result += ",".join(
                    [
                        "{}".format(getattr(self[jdx], subtype[:-1].lower()))
                        for jdx in jdxs
                    ]
                )
            else:
                result += ",".join(["{}".format(i) for i in np.arange(len(jdxs))])
            result += "\n"
        return result

    def _safeGetScalarAttr(self, attrName):
        # return np.nan when the attribute is missing, so that the returned value can be used in a comparison
        # e.g., lcc[lcc.sector < 25]
        return np.array([getattr(lcOrTpf, attrName, np.nan) for lcOrTpf in self.data])

    @property
    def sector(self):
        """(TESS-specific) the sectors of the lightcurves / target pixel files.

        The TESS sectors of the lightcurves / target pixel files; `numpy.nan` for those with no sector.
        The attribute is useful for filtering a collection by sector.

        Examples
        --------
        Plot two lightcurves, one from TESS sectors 13 to 19, and one for sector 22

            >>> import lightkurve as lk
            >>> lcc = lk.search_lightcurve('TIC286923464', mission='TESS').download_all()  # doctest: +SKIP
            >>> lcc_filtered = lcc[(lcc.sector >= 13) & (lcc.sector <= 19)]  # doctest: +SKIP
            >>> lcc_filtered.plot()  # doctest: +SKIP
            >>> lcc[lcc.sector == 22][0].plot()  # doctest: +SKIP

        """
        return self._safeGetScalarAttr("sector")

    @property
    def quarter(self):
        """(Kepler-specific) the quarters of the lightcurves / target pixel files.

        The Kepler quarters of the lightcurves / target pixel files; `numpy.nan` for those with none.
        """
        return self._safeGetScalarAttr("quarter")

    @property
    def campaign(self):
        """(K2-specific) the campaigns of the lightcurves / target pixel files.

        The K2 campaigns of the lightcurves / target pixel files; `numpy.nan` for those with none.
        """
        return self._safeGetScalarAttr("campaign")


class LightCurveCollection(Collection):
    """Class to hold a collection of LightCurve objects.

    Attributes
    ----------
    lightcurves : array-like
        List of LightCurve objects.
    """

    def __init__(self, lightcurves):
        super(LightCurveCollection, self).__init__(lightcurves)

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def PDCSAP_FLUX(self):
        """DEPRECATED. Replaces `LightCurveFileCollection.PDCSAP_FLUX`.
        Provided for backwards-compatibility with Lightkurve v1.x;
        will be removed soon."""
        return LightCurveCollection([lc.PDCSAP_FLUX for lc in self])

    @property
    @deprecated("2.0", warning_type=LightkurveDeprecationWarning)
    def SAP_FLUX(self):
        """DEPRECATED. Replaces `LightCurveFileCollection.SAP_FLUX`.
        Provided for backwards-compatibility with Lightkurve v1.x;
        will be removed soon."""
        return LightCurveCollection([lc.SAP_FLUX for lc in self])

    def stitch(self, corrector_func=lambda x: x.normalize()):
        """Stitch all light curves in the collection into a single lk.LightCurve

        Any function passed to `corrector_func` will be applied to each light curve
        before stitching. For example, passing "lambda x: x.normalize().flatten()"
        will normalize and flatten each light curve before stitching.

        Parameters
        ----------
        corrector_func : function
            Function that accepts and returns a `~lightkurve.lightcurve.LightCurve`.
            This function is applied to each light curve in the collection
            prior to stitching. The default is to normalize each light curve.

        Returns
        -------
        lc : `~lightkurve.lightcurve.LightCurve`
            Stitched light curve.
        """
        if corrector_func is None:
            corrector_func = lambda x: x  # noqa: E731
        with warnings.catch_warnings():  # ignore "already normalized" message
            warnings.filterwarnings("ignore", message=".*already.*")
            lcs = [corrector_func(lc) for lc in self]
        # Need `join_type='inner'` until AstroPy supports masked Quantities
        return vstack(lcs, join_type="inner", metadata_conflicts="silent")

    def plot(self, ax=None, offset=0.0, **kwargs) -> matplotlib.axes.Axes:
        """Plots all light curves in the collection on a single plot.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.
        offset : float
            Offset to add to targets with different labels, to prevent light
            curves from being plotted on top of each other.  For example, if
            the collection contains light curves with unique labels "A", "B",
            and "C", light curves "A" will have `0*offset` added to their flux,
            light curves "B" will have `1*offset` offset added, and "C" will
            have `2*offset` added.
        **kwargs : dict
            Dictionary of arguments to be passed to `LightCurve.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        with plt.style.context(MPLSTYLE):
            if ax is None:
                _, ax = plt.subplots()
            for kwarg in ["c", "color", "label"]:
                if kwarg in kwargs:
                    kwargs.pop(kwarg)

            labels = np.asarray([lc.meta.get("LABEL") for lc in self])
            try:
                unique_labels = np.sort(np.unique(labels))
            except TypeError:  # sorting will fail if labels includes None
                unique_labels = [None]

            for idx, targetid in enumerate(unique_labels):
                jdxs = np.where(labels == targetid)[0]
                for jdx in np.atleast_1d(jdxs):
                    if jdx != jdxs[0]:  # Avoid multiple labels for same object
                        kwargs["label"] = ""
                    self[jdx].plot(ax=ax, c=f"C{idx}", offset=idx * offset, **kwargs)
        return ax


class TargetPixelFileCollection(Collection):
    """Class to hold a collection of `~lightkurve.targetpixelfile.TargetPixelFile` objects.

    Parameters
    ----------
    tpfs : list or iterable
        List of `~lightkurve.targetpixelfile.TargetPixelFile` objects.
    """

    def __init__(self, tpfs):
        super(TargetPixelFileCollection, self).__init__(tpfs)

    def plot(self, ax=None):
        """Individually plots all TargetPixelFile objects in a single
        matplotlib axes object.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be created.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if ax is None:
            _, ax = plt.subplots(len(self.data), 1, figsize=(7, (7 * len(self.data))))
        if len(self.data) == 1:
            self.data[0].plot(ax=ax)
        else:
            for i, tpf in enumerate(self.data):
                tpf.plot(ax=ax[i])
        return ax
