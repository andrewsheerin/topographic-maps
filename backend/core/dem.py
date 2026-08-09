"""DEM acquisition and preparation: download from OpenTopography, clip to the
polygon, reproject to UTM, and read into a numpy array.

CRS handling is explicit throughout (see science-integrity: CRS is never
assumed, nodata is never silently filled)."""

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
import requests
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine

import config
import constants


def _normalize_dataset(name: str) -> str:
    if not name:
        return constants.DEFAULT_DATASET
    name = str(name).strip()
    return constants.DATASET_ALIASES.get(name, name)


def download_dem(w, s, e, n, out_path, dataset="USGS10m"):
    """Download a GeoTIFF DEM for the bounds (w, s, e, n) in WGS84 degrees.

    USGS products come from /API/usgsdem; all others from /API/globaldem.
    """
    api_key = config.require_opentopo_api_key()
    dataset = _normalize_dataset(dataset)

    if dataset in constants.USGSDEM_DATASETS:
        url = "https://portal.opentopography.org/API/usgsdem"
        params = dict(
            datasetName=dataset,
            west=w,
            south=s,
            east=e,
            north=n,
            outputFormat="GTiff",
            API_Key=api_key,
        )
        r = requests.get(url, params=params, timeout=180)
        if not r.ok:
            raise RuntimeError(
                "OpenTopography USGSDEM request failed "
                f"(HTTP {r.status_code}) for datasetName='{dataset}'. Response: {r.text.strip()}"
            )
    else:
        url = "https://portal.opentopography.org/API/globaldem"
        params = dict(
            demtype=dataset,
            west=w,
            south=s,
            east=e,
            north=n,
            outputFormat="GTiff",
            API_Key=api_key,
        )
        r = requests.get(url, params=params, timeout=180)
        if not r.ok:
            raise RuntimeError(
                "OpenTopography GlobalDEM request failed "
                f"(HTTP {r.status_code}) for demtype='{dataset}'. Response: {r.text.strip()}"
            )

    with open(out_path, "wb") as f:
        f.write(r.content)


def clip_dem_by_polygon(dem_path, polygon, crs, out_path):
    """Crop a DEM to `polygon` (given in `crs`), reprojecting the polygon to the
    raster's CRS first."""
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    with rasterio.open(dem_path) as src:
        gdf = gdf.to_crs(src.crs)
        out, transform = rio_mask(src, gdf.geometry, crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": out.shape[1],
                "width": out.shape[2],
                "transform": transform,
            }
        )
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(out)


def reproject_dem(src_path, dst_path, epsg):
    """Reproject a DEM to the target EPSG, preserving nodata so it does not
    become a huge 'valid' elevation."""
    with rasterio.open(src_path) as src:
        dst_crs = CRS.from_epsg(epsg)
        transform, w, h = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()

        src_nodata = src.nodata
        dst_nodata = src_nodata if src_nodata is not None else -9999.0

        meta.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": w,
                "height": h,
                "nodata": dst_nodata,
            }
        )

        with rasterio.open(dst_path, "w", **meta) as dst:
            rasterio.warp.reproject(
                rasterio.band(src, 1),
                rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=dst_nodata,
                resampling=Resampling.bilinear,
            )


def read_dem(path, downsample):
    """Read band 1 as float32 metres, optionally averaging-downsampled. Nodata
    is masked to NaN (never silently filled). Returns (data, px_m, transform)."""
    with rasterio.open(path) as src:
        transform = src.transform
        nodata = src.nodata

        if downsample > 1:
            data = src.read(
                1,
                out_shape=(
                    max(1, src.height // downsample),
                    max(1, src.width // downsample),
                ),
                resampling=Resampling.average,
            )
            transform = transform * Affine.scale(downsample)
        else:
            data = src.read(1)

        px_m = abs(transform.a)

    data = data.astype(np.float32)

    if nodata is not None and np.isfinite(nodata):
        data[data == np.float32(nodata)] = np.nan

    data[~np.isfinite(data)] = np.nan
    return data, px_m, transform
