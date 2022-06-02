from typing import Dict, List, Tuple

from shapely.geometry import Polygon, JOIN_STYLE
from shapely.strtree import STRtree

JOIN_TOLERANCE = 0.001


def find_intersections(
    polygons: List[Polygon],
) -> Dict[Tuple[int, int], Polygon]:
    tree = STRtree(polygons)
    tree_map = {id(x): i for i, x in enumerate(polygons)}
    intersections = {}
    for i, x in enumerate(polygons):
        res = tree.query(x)
        for y in res:
            j = tree_map[id(y)]
            if i == j:
                continue
            key = tuple(sorted((i, j)))
            if key not in intersections:
                intersection = x.intersection(y)
                # Sometimes intersections contain small "slivers" of polygons with almost zero
                # area. Removing them
                # https://gis.stackexchange.com/a/120314
                intersection_fixed = intersection.buffer(
                    JOIN_TOLERANCE, 1, join_style=JOIN_STYLE.mitre
                ).buffer(-JOIN_TOLERANCE, 1, join_style=JOIN_STYLE.mitre)
                # Only add the intersection if it is not empty
                intersections[key] = (
                    intersection_fixed if intersection_fixed.area > 1 else None
                )
    # Remove empty intersections
    for k in list(intersections.keys()):
        if intersections[k] is None:
            del intersections[k]
    return intersections
