from typing import Tuple

import pandas as pd
import tifffile
import zarr

from .tide import ContactProfiler


def process_tiffs(
    intensity_path: str,
    label_path: str,
    max_radius: int = 3,
    cache_size: int = 10 * 1024**3,
    return_profiler: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, ContactProfiler]:
    intensity_tiff = tifffile.TiffFile(intensity_path, mode="rb")
    label_tiff = tifffile.TiffFile(label_path, mode="rb")
    cache_ratio = intensity_tiff.series[0].nbytes / label_tiff.series[0].nbytes
    intensity_cache_size = int(cache_size * cache_ratio / (1 + cache_ratio))
    intensity_image = zarr.open(
        zarr.LRUStoreCache(intensity_tiff.aszarr(), max_size=intensity_cache_size)
    )[0]
    label_image = zarr.open(
        zarr.LRUStoreCache(label_tiff.aszarr(), max_size=cache_size - intensity_cache_size)
    )[0]
    profiler = ContactProfiler(label_image, intensity_image)
    df = profiler.contact_profile_table(max_radius)
    if return_profiler:
        return df, profiler
    else:
        return df
