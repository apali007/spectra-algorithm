# ==============
# Filters a generated reasoning-corpus JSON down to a single case_label.
#
# Useful for building a case-specific training set (e.g. "clean y-ion ladder
# only") from the full, all-cases corpus, without regenerating anything.
#
# Usage:
#   python scripts/filter_case_type.py \
#       --input output/mass_analyzer_aware_prospect_reasoning_dataset.json \
#       --case_label CASE_CLEAN_Y_ION_LADDER \
#       --output output/clean_y_ion_only_corrected.json

import argparse
import json
from pathlib import Path
from collections import Counter

DEFAULT_INPUT = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/mass_analyzer_aware_prospect_reasoning_dataset.json'
DEFAULT_CASE_LABEL = 'CASE_CLEAN_Y_ION_LADDER'
DEFAULT_OUTPUT = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/clean_y_ion_only_full_mass_analyzer_aware_prospect_reasoning_dataset.json'


def main():
    parser = argparse.ArgumentParser(description="Filter a reasoning corpus down to a single case_label")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help="Path to the generated reasoning-corpus JSON to filter.")
    parser.add_argument("--case_label", default=DEFAULT_CASE_LABEL,
                         help="case_label to keep. Everything else is dropped.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                         help="Path to write the filtered JSON to.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as f:
        dataset = json.load(f)

    print(f"Input: {input_path}")
    print(f"Total examples: {len(dataset)}")

    all_counts = Counter(item.get('case_label', 'UNKNOWN') for item in dataset)
    print("\nFull case distribution:")
    for label in sorted(all_counts):
        print(f" {label}: {all_counts[label]} ({100*all_counts[label]/len(dataset):.1f}%)")

    filtered = [item for item in dataset if item.get('case_label') == args.case_label]

    print(f"\nFiltered to case_label = {args.case_label}")
    print(f"Kept: {len(filtered)} of {len(dataset)} ({100*len(filtered)/len(dataset):.1f}%)")

    with open(output_path, 'w') as f:
        json.dump(filtered, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()