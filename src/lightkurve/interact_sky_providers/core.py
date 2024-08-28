from abc import ABC, abstractmethod
from collections import namedtuple

from typing import Tuple, Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table

__all__ = ["InteractSkyCatalogProvider", "ProperMotionCorrectionMeta"]


ProperMotionCorrectionMeta = namedtuple(
    "ProperMotionCorrectionMeta", "ra_colname dec_colname pmra_colname pmdec_colname frame equinox"
)


class InteractSkyCatalogProvider(ABC):
    """Abstract class for providing catalog data to
    `TargetPixelFile.interact_sky() <lightkurve.TargetPixelFile.interact_sky>`.

    A subclass must specify the following attributes (typically in the constructor).

    Parameters
    ----------
    cols_for_source : list
        The list of column names (of the query result table) to be included
        in the data source.

    cols_as_str_for_source : list
        The list of column names (of the query result table) of which the values
        are to be converted to string when being included in the data source.
        Typically it is used to workaround the bokeh issue of transmitting 64-bit integers
        in a data source.
    """

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
    ) -> None:
        """Constructor.

        A subclass should define ``scatter_kwargs`` to distinguish its data
        from other providers.

        """
        # query: generic
        self.coord = coord
        self.radius = radius
        self.magnitude_limit = magnitude_limit
        # for plotting
        if scatter_kwargs is None:
            scatter_kwargs = dict(  # some default rendering that should be overridden
                marker="circle",
                fill_alpha=0.3,
                fill_color="red",
            )
        self.scatter_kwargs = scatter_kwargs

        # (extra) columns to be included in bokeh data source
        # interact_sky() logic always adds the following to the data source:
        # - ra, dec, x, y, separation, size
        self.cols_for_source = []
        # columns to be converted to string in bokeh data source
        # (primarily to workaround bokeh issue of handling large integers)
        self.cols_as_str_for_source = []

    @property
    @abstractmethod
    def label(self) -> str:
        """Label to identify the catalog in UI."""
        pass

    @abstractmethod
    def query_catalog(self) -> Table:
        """A subclass would generally implement the logic to search the remote catalog.

        The return table should have columns ``RA`` and ``DEC`` for
        proper-motion corrected positions.
        Alternatively, the position / proper motion column names can be specified
        with ``get_proper_motion_correction_meta()``.

        In addition, the return table must have column ``magForSize``, which reflects
        the magnitude of the target for the purpose of scaling in the plot.
        """
        pass

    @abstractmethod
    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        """Return ``None`` for tables with columns ``RA`` and ``DEC``
        (proper motion corrected positions), or a `ProperMotionCorrectionMeta` object
        to specify the columns for positions and proper motions.
        """
        pass

    def add_to_data_source(self, result: Table, source: dict):
        """Convert query result table to a dictionary, which will be wrapped as
        bokeh ColumnDataSource.

        The behavior is specified by attributes `cols_for_source`
        and `cols_as_str_for_source`. A subclass generally does not need to override it.
        """
        more_data = dict()
        for col in self.cols_for_source:
            # bokeh ColumnDataSource-specific workaround: convert some columns to string
            # usually it is for columns with large integers, to avoid
            # BokehUserWarning: out of range integer may result in loss of precision
            col_val = result[col]
            if col in self.cols_as_str_for_source:
                col_val = col_val.astype(str)
            more_data[col] = col_val
        source.update(more_data)

    @abstractmethod
    def get_tooltips(self) -> list:
        """Return the bokeh on hover tooltips.

        See: https://docs.bokeh.org/en/latest/docs/reference/models/tools.html#bokeh.models.HoverTool
        """
        pass

    @abstractmethod
    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        """Return the data to render the detail view upon clicking a star.

        Parameters
        ----------
        data: dict
            the star's data in the catalog (from bokeh data source)

        Returns
        -------
        detail_view : (dict, list)
            The data to be rendered in the detail view.
            The dict contains header - value pairs. The values can include HTML formatting.
            The list contains additional (HTML-formatted) data to be rendered (typically as rows).
            The list can be ``None`` if no extra information is needed.
        """
        pass
