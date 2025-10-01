from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out_path = root / "data/sample/dem/synthetic_world_dem.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Global grid at 1 degree resolution in EPSG:4326
    width, height = 360, 180
    pixel_size = 1.0
    # origin at upper-left corner (-180, 90)
    transform = from_origin(-180.0, 90.0, pixel_size, pixel_size)

    # Create a simple sloped DEM with gentle variation
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    dem = 3000.0 * (1.0 - 0.6 * y) + 500.0 * x
    dem = dem.astype(np.float32)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "crs": CRS.from_epsg(4326),
        "transform": transform,
        "nodata": None,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dem, 1)

    print(f"Synthetic DEM written: {out_path}")


if __name__ == "__main__":
    main()