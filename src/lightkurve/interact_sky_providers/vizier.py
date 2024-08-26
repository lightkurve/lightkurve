from typing import Union
import warnings

import astropy.units as u

from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.vizier import Vizier

from .core import InteractSkyCatalogProvider


def _query_cone_region(coord, radius, catalog, columns=["*"], query_kwargs=dict()) -> Table:
    # Thin wrapper over Vizier's query_region
    vizier = Vizier(
        columns=columns,
    )
    vizier.ROW_LIMIT = -1
    return vizier.query_region(coord, radius=radius, catalog=catalog, **query_kwargs)


class VizierInteractSkyCatalogProvider(InteractSkyCatalogProvider):

    def __init__(
        self,
        coord: SkyCoord,
        radius: Union[float, u.Quantity],
        magnitude_limit: float,
        scatter_kwargs: dict = None,
    ) -> None:
        super().__init__(coord, radius, magnitude_limit, scatter_kwargs)
        # Vizier-specific query
        self.catalog_name = None
        self.columns = ["*"]
        self.magnitude_limit_column_name = None

    def query_catalog(self) -> Table:
        with warnings.catch_warnings():
            # suppress useless warning to workaround  https://github.com/astropy/astroquery/issues/2352
            # for Gaia
            warnings.filterwarnings("ignore", category=u.UnitsWarning, message="Unit 'e' not supported by the VOUnit standard")
            result = _query_cone_region(self.coord, self.radius, self.catalog_name, columns=self.columns)
        if result is None or len(result) == 0:
            return None
        result = result[self.catalog_name]
        if self.magnitude_limit_column_name is not None and self.magnitude_limit is not None:
            result = result[result[self.magnitude_limit_column_name] < self.magnitude_limit]
        # to be used as the basis for sizing the dots in plots
        if self.magnitude_limit_column_name is not None:
            result["magForSize"] = result[self.magnitude_limit_column_name]

        return result
