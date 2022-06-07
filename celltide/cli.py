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
    )
    contact_profiles.to_csv(
        os.path.join(args.OUTPUT_DIR, "contact_profiles.csv"), index=False
    )
    cell_profiles = profiler.radial_regionprops(
        list(range(-args.max_radius, args.max_radius + 1))
    )
    cell_profile_df = pd.concat(
        [pd.DataFrame.from_dict(x) for x in cell_profiles.values()],
        keys=cell_profiles.keys(),
        names=["expansion_radius", "row_id"],
    )
    cell_profile_df.reset_index(inplace=True)
    cell_profile_df.to_csv(
        os.path.join(args.OUTPUT_DIR, "cell_profiles.csv"), index=False
    )
    logging.info(
        f"Intensity image cache hits {profiler.intensity_image.store.hits} misses {profiler.intensity_image.store.misses} \n"
    )
    logging.info(
        f"Label image cache hits {profiler.label_image.store.hits} misses {profiler.label_image.store.misses}"
    )
    logging.info("Done")
    import pdb
    pdb.set_trace()
