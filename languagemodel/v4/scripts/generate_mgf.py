"""
Convert clean_y_ion test.json entries into an annotated MGF spectra file for
Casanovo evaluation, using the cached-spectra and joining
in retention time from the meta_data parquet

Also, the cache data and meta_data parquest have one to one mapping on the key (peptide, scan_number, raw_file)
"""
import argparse
import json
import logging
import pickle

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def build_argparser():
    parser = argparse.ArgumentParser(description="Convert test.json to annotated MGF using cached spectra + meta_data retention time")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test.json")
    parser.add_argument("--cache", type=str, required=True, help="Path to the cache .pkl file")
    parser.add_argument("--meta_data", type=str, required=True, help="Path to TUM_third_pool_meta_data.parquet")
    parser.add_argument("--output", type=str, required=True, help="Path to write the .mgf file")
    return parser

def main():
    parser = build_argparser()
    args = parser.parse_args()

    logger.info(f"Loading cache: {args.cache}")
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)

    logger.info(f"Building spectra lookup from {len(cache)} cached spectra")
    spectra_lookup = {
        (entry['scan_number'], entry['raw_file'], entry['peptide']): entry 
        for entry in cache
    }

    logger.info(f"Loading meta_data: {args.meta_data}")
    meta = pd.read_parquet(args.meta_data)
    logger.info(f"Building retention-time lookup from {len(meta)} meta_data rows")
    rt_lookup = {
        (row.scan_number, row.raw_file, row.modified_sequence): row.retention_time
        for row in meta.itertuples(index=False)
    }

    logger.info(f"Loading test set: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test examples")

    written = 0
    missing_spectra = 0
    missing_rt = 0

    with open(args.output, 'w') as out_f:
        for i, item in enumerate(test_data, 1):
            scan_number = item.get('scan_number')
            raw_file = item.get('raw_file')
            peptide = item.get('peptide')

            entry = spectra_lookup.get((scan_number, raw_file, peptide))
            if entry is None:
                missing_spectra += 1
                continue
            
            rt_minutes = rt_lookup.get((scan_number, raw_file, peptide))
            if rt_minutes is None:
                missing_rt += 1
                rt_seconds = None
            else:
                rt_seconds = rt_minutes * 60
            

            precursor_mz = entry['precursor_mz']
            charge = entry['precursor_charge']
            annotation_rows = entry['annotation_rows']
            mz_list = annotation_rows['experimental_mass'].tolist()
            intensity_list = annotation_rows['intensity'].tolist()

            unique_peaks = sorted(set(zip(mz_list, intensity_list)), key=lambda p: p[0])

            out_f.write("BEGIN IONS\n")
            out_f.write(f"TITLE={i}\n")
            out_f.write(f"PEPMASS={precursor_mz}\n")
            out_f.write(f"CHARGE={charge}+\n")
            out_f.write(f"SCANS={scan_number}\n")
            if rt_seconds is not None:
                out_f.write(f"RTINSECONDS={rt_seconds}\n")
            out_f.write(f"SEQ={peptide}\n")
            for mz, intensity in unique_peaks:
                out_f.write(f"{mz} {intensity}\n")
            out_f.write("END IONS\n\n")
            written += 1

    logger.info(f"Wrote {written} spectra to {args.output}")
    if missing_spectra:
        logger.warning(f"{missing_spectra} test examples had no matching cache entry - skipped")
    if missing_rt:
        logger.warning(f"{missing_rt} written spectra had no retention-time match - RTINSECONDS ommitted for those")


if __name__ == "__main__":
    main()
