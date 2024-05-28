from abc import ABC, abstractmethod
from collections import namedtuple

from typing import Tuple, Union

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table

__all__ = ["InteractSkyCatalogProvider", "ProperMotionCorrectionMeta"]


ProperMotionCorrectionMeta = namedtuple(
    "ProperMotionCorrectionMeta", "ra_colname dec_colname pmra_colname pmdec_colname equinox"
)


class InteractSkyCatalogProvider(ABC):
    # query: generic
    coord: SkyCoord
    radius: Union[float, u.Quantity]
    magnitude_limit: float
    # for plotting
    scatter_kwargs: dict
    extra_cols_for_source = []  # extra columns to be included in bokeh data source
    extra_cols_as_str_for_source = []  # columns to be converted to string in bokeh data source
    label: str = None

    def init(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict,
    ) -> None:
        self.coord = coord
        self.radius = radius
        self.magnitude_limit = magnitude_limit
        if scatter_kwargs is None:
            scatter_kwargs = dict(  # some default rendering that should be overridden
                marker="circle",
                fill_alpha=0.3,
                fill_color="red",
            )
        self.scatter_kwargs = scatter_kwargs

    @abstractmethod
    def query_catalog(self) -> Table:
        pass

    @abstractmethod
    def get_proper_motion_correction_meta(self) -> ProperMotionCorrectionMeta:
        pass

    def add_to_data_source(self, result: Table, source: dict):
        more_data = dict()
        for col in self.extra_cols_for_source:
            # bokeh ColumnDataSource-specific workaround: convert some columns to string
            # usually it is for columns with large integers, to avoid
            # BokehUserWarning: out of range integer may result in loss of precision
            col_val = result[col]
            if col in self.extra_cols_as_str_for_source:
                col_val = col_val.astype(str)
            more_data[col] = col_val
        source.update(more_data)

    @abstractmethod
    def get_tooltips(self) -> list:
        pass

    @abstractmethod
    def get_detail_view(self, data: dict) -> Tuple[dict, list]:
        pass
