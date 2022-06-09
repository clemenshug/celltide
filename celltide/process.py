import logging
from typing import Tuple

import numpy as np
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
    store_shapes: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, ContactProfiler]:
    intensity_tiff = tifffile.TiffFile(intensity_path, mode="rb")
    label_tiff = tifffile.TiffFile(label_path, mode="rb")
    total_image_bytes = sum(x.series[0].nbytes for x in (intensity_tiff, label_tiff))
    logging.info(f"Total image size: {total_image_bytes / 1024**3:.1f} GB")
    if cache_size > total_image_bytes:
        logging.info("Cache size is larger than total image size. Loading images into memory.")
        intensity_image = intensity_tiff.series[0].asarray()
        label_image = label_tiff.series[0].asarray()
    else:
        logging.info("Cache size is smaller than total image size. Caching only most recent slices.")
        cache_ratio = intensity_tiff.series[0].nbytes / label_tiff.series[0].nbytes
        intensity_cache_size = int(cache_size * cache_ratio / (1 + cache_ratio))
        intensity_image = zarr.open(
            zarr.LRUStoreCache(intensity_tiff.aszarr(), max_size=intensity_cache_size)
        )
        if isinstance(intensity_image, zarr.Group):
            intensity_image = intensity_image[0]
        label_image = zarr.open(
            zarr.LRUStoreCache(
                label_tiff.aszarr(), max_size=cache_size - intensity_cache_size
            )
        )
        if isinstance(label_image, zarr.Group):
            label_image = label_image[0]
    profiler = ContactProfiler(
        label_image,
        intensity_image,
        store_profile_masks=store_shapes,
        store_annuli=store_shapes,
    )
    df = profiler.contact_profile_table(max_radius)
    if return_profiler:
        return df, profiler
    else:
        return df
