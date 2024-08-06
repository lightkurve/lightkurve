"""Defines collections of data products."""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import vstack
from astropy.utils.decorators import deprecated

from . import MPLSTYLE
from .utils import LightkurveWarning, LightkurveDeprecationWarning


__all__ = ["LightCurveCollection", "TargetPixelFileCollection"]


class Collection(object):
    """Base class for `LightCurveCollection` and `TargetPixelFileCollection`.

    A collection can be indexed by standard Python list syntax.
    Additionally, it can be indexed by a subset of `numpy.ndarray` syntax:
    boolean array indexing and integer array indexing.

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
        if isinstance(index_or_mask, (int, np.integer)):
            return self.data[index_or_mask]
        elif isinstance(index_or_mask, slice):
            return type(self)(self.data[index_or_mask])
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
        result = f"{self.__class__.__name__} of {len(self)} objects:\n    "
        # LightCurve objects provide a special `_repr_simple_` method
        # to avoid printing an entire table here
        result += "\n    ".join(
            [
                f"{idx}: " + getattr(obj, "_repr_simple_", obj.__repr__)()
                for idx, obj in enumerate(self)
            ]
        )
        return result

    def _safeGetScalarAttr(self, attrName):
        # return np.nan when the attribute is missing, so that the returned value can be used in a comparison
        # e.g., lcc[lcc.sector < 25]
        return np.array([getattr(lcOrTpf, attrName, np.nan) for lcOrTpf in self.data])

    @property
    def sector(self):
        """(TESS-specific) the quarters of the lightcurves / target pixel files.

        Returns `numpy.nan` for data products with lack a sector meta data keyword.
        The attribute is useful for filtering a collection by sector.

        Examples
        --------
        Plot two lightcurves, one from TESS sectors 13 to 19, and one for sector 22.

            >>> import lightkurve as lk
            >>> lcc = lk.search_lightcurve('TIC286923464', author='SPOC').download_all()  # doctest: +SKIP
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
        """Stitch all light curves in the collection into a single `LightCurve`.

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

        # Address issue #954: ignore incompatible columns with the same name
        columns_to_remove = set()
        for col in lcs[0].columns:
            for lc in lcs[1:]:
                if col in lc.columns:
                    if not (
                        issubclass(lcs[0][col].__class__, lc[col].__class__)
                        or issubclass(lc[col].__class__, lcs[0][col].__class__)
                        or lcs[0][col].__class__.info is lc[col].__class__.info
                    ):
                        columns_to_remove.add(col)
                        continue

        if len(columns_to_remove) > 0:
            warnings.warn(
                f"The following columns will be excluded from stitching because the column types are incompatible: {columns_to_remove}",
                LightkurveWarning,
            )
            lcs = [lc.copy() for lc in lcs]
            [
                lc.remove_columns(columns_to_remove.intersection(lc.columns))
                for lc in lcs
            ]

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

            for idx, lc in enumerate(self):
                kwargs["label"] = f"{idx}: {lc.meta.get('LABEL', '(missing label)')}"
                lc.plot(ax=ax, c=f"C{idx}", offset=idx * offset, **kwargs)

            # If some but not all light curves are normalized, ensure the Y label
            # says "Flux" and not "Normalized Flux"
            normstatus = [lc.meta.get("NORMALIZED", False) for lc in self]
            if "normalize" not in kwargs and any(normstatus) and not all(normstatus):
                warnings.warn(
                    "Some but not all of the light curves in the collection appear to be normalized. "
                    "You may wish to use `normalize=True` to ensure all are normalized.",
                    LightkurveWarning,
                )
                if "ylabel" not in kwargs:
                    ax.set_ylabel("Flux")

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
