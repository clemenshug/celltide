import logging
from collections import defaultdict
from itertools import pairwise
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy import ndimage
from skimage import measure, morphology
from skimage.measure._regionprops import RegionProperties, _props_to_dict

from .intersections import find_intersections
from .utils import ImageType


class LazyExpandedRegions:
    def __init__(self, profiler: "ContactProfiler", radius: int):
        self.radius = radius
        self.profiler = profiler
        self.label_image = profiler.label_image

    def __getitem__(self, x: int) -> RegionProperties:
        if x % 10000 == 0:
            logging.info(f"Expanding region {x}")
        label = x + 1
        sl_expanded, label_image_crop = self.profiler.get_expanded_cell_mask(
            x, self.radius, return_bool=False
        )
        intensity_image_crop = np.moveaxis(
            self.profiler.intensity_image[:, sl_expanded[0], sl_expanded[1]], 0, -1
        )
        return RegionProperties(
            slice(None),
            label,
            label_image_crop,
            intensity_image_crop,
            cache_active=True,
            extra_properties=self.profiler.extra_properties,
        )

    def __len__(self):
        return len(self.profiler.objects)


class ContactProfiler:
    def __init__(
        self,
        label_image: ImageType,
        intensity_image: ImageType,
        extra_properties=None,
        store_profile_masks: bool = False,
        store_annuli: bool = False,
    ):
        self.label_image = label_image
        self.intensity_image = intensity_image
        self.objects = ndimage.find_objects(label_image)
        self.extra_properties = extra_properties
        self.store_profile_masks = store_profile_masks
        self.store_annuli = store_annuli
        self._footprints = {}
        self.intersections = None
        self.annuli = {}
        self.profile_masks = {}
        self.contours = self.find_contours(1)

    def _get_footprint(self, radius: int):
        try:
            return self._footprints[radius]
        except KeyError:
            self._footprints[radius] = morphology.disk(radius)
            return self._footprints[radius]

    def find_contours(self, radius: int) -> List[np.ndarray]:
        contours = []
        for i in range(len(self.objects)):
            sl, mask = self.get_expanded_cell_mask(i, radius)
            mask = ndimage.binary_fill_holes(mask)
            contour = np.concatenate(
                measure.find_contours(mask, level=0.5, fully_connected="high"), axis=0
            )
            contour += np.array([x.start for x in sl])
            contours.append(contour)
        return contours

    def find_intersections(self):
        polygons = [Polygon(x) for x in self.contours if x.shape[0] >= 3]
        intersections = find_intersections(polygons)
        self.intersections = intersections
        return intersections

    def get_expanded_cell_mask(
        self,
        x: int,
        radius: int,
        crop_slice: Tuple[slice, ...] = None,
        return_bool: bool = True,
    ) -> Tuple[Tuple[slice, ...], np.ndarray]:
        label = x + 1
        if crop_slice is None:
            sl = self.objects[x]
            crop_slice = tuple(
                slice(
                    max(0, x.start - radius - 1),
                    min(self.label_image.shape[i], x.stop + radius + 1),
                )
                for i, x in enumerate(sl)
            )
        crop = self.label_image[crop_slice[0], crop_slice[1]] == label
        if radius > 0:
            crop = morphology.binary_dilation(
                crop, footprint=self._get_footprint(radius)
            )
        elif radius < 0:
            crop = morphology.binary_erosion(
                crop, footprint=self._get_footprint(abs(radius))
            )
        if not return_bool:
            crop = crop.astype(self.label_image.dtype)
            crop *= label
        return crop_slice, crop

    def radial_regionprops(
        self, radii: Iterable[int], **kwargs
    ) -> Dict[int, Dict[str, np.ndarray]]:
        prop_tables = {}
        for r in radii:
            extra_properties = kwargs.get("extra_properties", None)
            properties = kwargs.get("properties", ("label", "area", "intensity_mean"))
            if extra_properties is not None:
                properties = list(properties) + [x.__name__ for x in extra_properties]
            prop_tables[r] = _props_to_dict(
                LazyExpandedRegions(self, r),
                properties=properties,
                separator=kwargs.get("separator", "-"),
            )
        return prop_tables

    def contact_profile(
        self, cell_pair: Tuple[int, int], max_radius: int
    ) -> np.ndarray:
        sl1, sl2 = self.objects[cell_pair[0]], self.objects[cell_pair[1]]
        target_slice = (
            slice(
                min(sl1[0].start, sl2[0].start) - max_radius,
                max(sl1[0].stop, sl2[0].stop) + max_radius,
            ),
            slice(
                min(sl1[1].start, sl2[1].start) - max_radius,
                max(sl1[1].stop, sl2[1].stop) + max_radius,
            ),
        )
        cell_masks = defaultdict(list)
        for i in range(2):
            for j in range(0, max_radius + 1):
                mask = self.get_expanded_cell_mask(
                    cell_pair[i], radius=j, crop_slice=target_slice
                )[1]
                cell_masks[i].append(mask)
        annuli = defaultdict(list)
        for i, masks in cell_masks.items():
            for m1, m2 in pairwise(reversed(masks)):
                annuli[i].append(np.bitwise_and(m1, ~m2))
        annuli[0].append(cell_masks[0][0])
        annuli[0].reverse()
        annuli[1].append(cell_masks[1][0])
        annuli[1].reverse()
        profile_masks = []
        for i in reversed(range(2, max_radius + 1)):
            profile_masks.append(np.bitwise_and(annuli[0][i], annuli[1][0]))
        profile_masks.append(np.bitwise_and(annuli[0][1], annuli[1][1]))
        for i in range(2, max_radius + 1):
            profile_masks.append(np.bitwise_and(annuli[1][i], annuli[0][0]))
        intensity_thumbnail = self.intensity_image[:, target_slice[0], target_slice[1]]
        signal_means = np.stack(
            [np.mean(intensity_thumbnail[:, p], axis=1) for p in profile_masks]
        )
        if self.store_profile_masks:
            self.profile_masks[cell_pair] = profile_masks
        if self.store_annuli:
            self.annuli[cell_pair] = annuli
        return signal_means

    def contact_profile_table(self, max_radius: int) -> pd.DataFrame:
        if self.intersections is None:
            self.find_intersections()
        profiles = {}
        for i, k in enumerate(self.intersections):
            if i % 1000 == 0:
                logging.info(
                    f"Processing cell contact {i} of {len(self.intersections)}"
                )
            profiles[k] = self.contact_profile(k, max_radius)

        def line_generator(x: Mapping[Tuple[int, int], np.ndarray]):
            for (c1, c2), p in x.items():
                for id_m in range(p.shape[1]):
                    yield (c1, c2, id_m) + tuple(p[:, id_m])

        profiles_df = pd.DataFrame.from_records(
            line_generator(profiles),
            columns=["cell_1", "cell_2", "marker_id"]
            + [
                ("-" if x < 0 else "+" if x > 0 else "") + str(abs(x))
                for x in range(1 - max_radius, max_radius)
            ],
        )
        return profiles_df
