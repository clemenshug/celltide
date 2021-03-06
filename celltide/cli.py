import argparse
import logging
import os
import pickle

import pandas as pd

from .process import process_tiffs


def celltide():
    parser = argparse.ArgumentParser(
        description="""""",
    )
    parser.add_argument(
        "IMAGE",
        help="Path to registered image in TIFF format",
    )
    parser.add_argument(
        "SEGMENTATION_MASK",
        help="Path to segmentation mask image in TIFF format",
    )
    parser.add_argument(
        "OUTPUT_DIR",
        help="Path to output directory. Existing files will only be overwritten "
        "if the --force flag is set.",
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action="store_true",
        help="Overwrite existing destination directory",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=3,
        help="Number of pixels that the contact profiles extend into cells. Default: 3",
    )
    parser.add_argument(
        "--cache-size",
        default=10 * 1024,
        type=int,
        help="Cache size for reading image tiles in MB. (Default: 10240 MB = 10 GB)",
    )
    parser.add_argument(
        "--store-shapes",
        action="store_true",
        help="Store shapes of pairwise cell-cell intersections and contact profiles "
        "in the output directory. ",
    )
    args = parser.parse_args()
    logging.basicConfig(
        format="%(processName)s %(asctime)s %(levelname)s: %(message)s",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logging.info(args)
    if (
        os.path.exists(args.OUTPUT_DIR)
        and len(os.listdir(args.OUTPUT_DIR)) > 0
        and not args.force
    ):
        parser.error("Output directory already exists. Use --force to overwrite.")
    try:
        os.mkdir(args.OUTPUT_DIR)
    except FileExistsError:
        pass
    contact_profiles, profiler = process_tiffs(
        args.IMAGE,
        args.SEGMENTATION_MASK,
        max_radius=args.max_radius,
        cache_size=args.cache_size * 1024**2,
        return_profiler=True,
        store_shapes=args.store_shapes,
    )
    contact_profiles.to_csv(
        os.path.join(args.OUTPUT_DIR, "contact_profiles.csv"), index=False
    )
    cell_profiles = profiler.radial_regionprops(
        list(range(-args.max_radius, args.max_radius + 1))
    )
    cell_profiles.to_csv(
        os.path.join(args.OUTPUT_DIR, "cell_profiles.csv"), index=False
    )
    logging.info(
        f"Intensity image cache hits {profiler.intensity_image.store.hits} misses {profiler.intensity_image.store.misses} \n"
    )
    logging.info(
        f"Label image cache hits {profiler.label_image.store.hits} misses {profiler.label_image.store.misses}"
    )
    if args.store_shapes:
        with open(os.path.join(args.OUTPUT_DIR, "annuli.pickle"), "wb") as f:
            pickle.dump(profiler.annuli, f)
        with open(os.path.join(args.OUTPUT_DIR, "profile_masks.pickle"), "wb") as f:
            pickle.dump(profiler.profile_masks, f)
        with open(os.path.join(args.OUTPUT_DIR, "cell_contours.pickle"), "wb") as f:
            pickle.dump(profiler.contours, f)
    logging.info("Done")
