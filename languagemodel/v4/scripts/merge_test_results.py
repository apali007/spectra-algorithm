'''
Merge sharded inference results files from run_inference.sh into one 
combined file, checking for spectra_key collisions across shards.
'''
import argparse
import glob
import json
import logging
import os
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

logger = logging.getLogger(__name__)

def spectra_key(item):
    raw_file = item.get('raw_file')
    scan_number = item.get('scan_number')
    peptide = item.get('peptide')
    return f"{raw_file}::{scan_number}::{peptide}"

def build_argsparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge sharded inference results into one file")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing shard*_results.json files, "
                        )
    parser.add_argument("--output", type=str, default=None,
                        help="Path for the merged results file. Defaults to " \
                        "<run_dir>/merged_results.json")

    return parser

def main():
    parser = build_argsparser()
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"--run_dir does not exist or it is not a directory: {args.run_dir}")

    shard_paths = sorted(glob.glob(os.path.join(args.run_dir, "shard*_results.json")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard*_results.json files found in {args.run_dir}")

    logger.info(f"Found {len(shard_paths)} shard files(s): {[os.path.basename(p) for p in shard_paths]}")

    merged = []
    for path in shard_paths:
        with open(path, 'r') as f:
            shard_results = json.load(f)
        logger.info(f"{os.path.basename(path)}: {len(shard_results)} spectra")
        merged.extend(shard_results)

    # Sanity check: sharding should produce disjoint keys across shards.
    key_counts = Counter(spectra_key(item) for item in merged)
    dupes = {key for key, count in key_counts.items() if count > 1}
    if dupes:
        logger.warning(f"{len(dupes)} duplicate spectra_key(s) found scross shards - " \
        f"these will be double-counted if evaluate as-is: {list(dupes)[:5]}"
        f"{'...' if len(dupes) > 5 else ''}"
        )

    logger.info(f"Total merged: {len(merged)} spectra")

    output_path = args.output or os.path.join(args.run_dir, "merged_results.json")
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    logger.info(f"Saved merged results to {output_path}")

    manifest_path = output_path.rsplit(".json", 1)[0] + "_manifest.json"
    manifest = {
        "run_dir": args.run_dir,
        "shard_files": [os.path.basename(p) for p in shard_paths],
        "total_spectra": len(merged),
        "duplicate_keys_found": len(dupes),
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved merge manifest to {manifest_path}")

if __name__ == "__main__":
    main()

