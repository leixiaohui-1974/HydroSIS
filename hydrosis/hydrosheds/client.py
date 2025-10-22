from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlretrieve


class HydroSHEDSClient:
    """Minimal client for accessing HydroSHEDS/HydroBASINS assets.

    This class focuses on caching and providing known download endpoints.
    Actual heavy processing occurs in the processor/pipeline modules.
    """

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, *parts: str) -> Path:
        path = self.cache_root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _download(self, url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, dest)
        return dest

    def download_wbd_subset(self, where: str, out_path: Path, bounding_box: Optional[Sequence[float]] = None) -> Path:
        """Download a WBD GeoJSON subset via USGS ArcGIS REST service.

        Parameters
        - where: ArcGIS SQL where clause, e.g. "AreaSqKm between 480 and 550".
        - out_path: target GeoJSON file.
        - bounding_box: optional [minx, miny, maxx, maxy] in EPSG:4326.
        """
        # Layer 6: HUC12 Subwatershed
        base = (
            "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/"
            "query?"
        )
        params = (
            f"where={where.replace(' ', '%20')}"
            "&outFields=*"
            "&f=geojson&returnGeometry=true"
        )
        if bounding_box and len(bounding_box) == 4:
            minx, miny, maxx, maxy = bounding_box
            params += (
                "&geometryType=esriGeometryEnvelope"
                "&spatialRel=esriSpatialRelIntersects"
                "&inSR=4326"
                f"&geometry={minx}%2C{miny}%2C{maxx}%2C{maxy}"
            )
        url = base + params
        return self._download(url, out_path)

    def note(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {message}")