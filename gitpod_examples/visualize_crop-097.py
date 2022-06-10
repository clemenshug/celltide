import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import tifffile

import celltide.tide

markers = pd.read_csv("CROP-097/markers.csv")
markers

intensity_img = tifffile.imread("CROP-097/registration/raw_sardana-097_crop.ome.tif", series=0)
label_image = tifffile.imread("CROP-097/segmentation/unmicst-raw_sardana-097_crop/cellMask.tif", series=0)

profiler = celltide.tide.ContactProfiler(
    label_image,
    intensity_img,
    store_annuli=True, store_profile_masks=True
)

intersections = profiler.find_intersections()

vis = profiler.get_visualizer()
vis.show_intensity_image()
vis.show_label_image()
vis.show_graph()
