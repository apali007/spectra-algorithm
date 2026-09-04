# ==============
# Train / validation / test split script for the prospect reasoning dataset.
#
# Peptide-disjoint split (default 80/10/10). All three sets are held disjoint
# by unique peptide (not just by spectrum) to avoid data leakage — the same
# peptide never appears in more than one of train/val/test, even though it
# may have many spectra.
#
# - train: used for weight updates
# - validation: used DURING training for checkpoint selection
#   (eval_strategy / load_best_model_at_end in train.py) — the model is
#   indirectly tuned against this set
# - test: held out completely, touched only once, at the end, for the
#   final reported numbers — never used for checkpoint selection
#
# Usage:
#   python scripts/train_test_split.py --input path/to/dataset.json --prefix my_run
#   (writes my_run_train.json, my_run_val.json, my_run_test.json into --output_dir)

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from collections import Counter

DEFAULT_INPUT = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_mass_analyzer_aware_prospect_reasoning_dataset.json'
DEFAULT_OUTPUT_DIR = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed'
DEFAULT_PREFIX = 'default_name'


def main():
    parser = argparse.ArgumentParser(description="Peptide-disjoint train/val/test split")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help="Path to the generated reasoning-corpus JSON to split.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write train/val/test JSON files into.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                         help="Filename prefix, e.g. 'prefix' -> prefix_train.json, prefix_val.json, prefix_test.json")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / f"{args.prefix}_train.json"
    val_output = output_dir / f"{args.prefix}_val.json"
    test_output = output_dir / f"{args.prefix}_test.json"

    with open(input_path, 'r') as f:
        dataset = json.load(f)

    print(f"Input: {input_path}")
    print(f"Total spectra in dataset: {len(dataset)}")

    peptide_groups = defaultdict(list)
    for entry in dataset:
        peptide_groups[entry['peptide']].append(entry)

    unique_peptides = list(peptide_groups.keys())
    random.seed(args.seed)
    random.shuffle(unique_peptides)

    n = len(unique_peptides)
    train_end = int(n * args.train_frac)
    val_end = int(n * (args.train_frac + args.val_frac))

    train_peptides = set(unique_peptides[:train_end])
    val_peptides = set(unique_peptides[train_end:val_end])
    test_peptides = set(unique_peptides[val_end:])

    train_set = [entry for p in train_peptides for entry in peptide_groups[p]]
    val_set = [entry for p in val_peptides for entry in peptide_groups[p]]
    test_set = [entry for p in test_peptides for entry in peptide_groups[p]]

    print(f"\nSpectra   - train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    print(f"Peptides  - train: {len(train_peptides)}, val: {len(val_peptides)}, test: {len(test_peptides)}")
    print(f"\nSplit fractions (by peptide) - train: {len(train_peptides)/n:.1%}, "
          f"val: {len(val_peptides)/n:.1%}, test: {len(test_peptides)/n:.1%}")

    show_case_stats("Train", train_set)
    show_case_stats("Val", val_set)
    show_case_stats("Test", test_set)

    with open(train_output, 'w') as f:
        json.dump(train_set, f, indent=2)
    with open(val_output, 'w') as f:
        json.dump(val_set, f, indent=2)
    with open(test_output, 'w') as f:
        json.dump(test_set, f, indent=2)

    print("\nDone.")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"  {test_output}")

def print_case_distribution(dataset, splits):
    print("\nCase distribution by split:")
    all_case_labels = sorted({entry.get("case_label", "UNKNOWN") for entry in dataset})
    header = f"{'case_label':<45}" + "".join(f"{name:>18}" for name in splits)
    print(header)
    for case_label in all_case_labels:
        row = f"{case_label:<45}"
        for name, split_data in splits.items():
            count = sum(1 for e in split_data if e.get("case_label", "UNKNOWN") == case_label)
            pct = 100 * count / len(split_data) if split_data else 0
            row += f"{count:>8} ({pct:>5.1f}%)"
        print(row)

def show_case_stats(label, data):
    total = len(data)
    case_counts = Counter(item.get('case_label', 'UNKNOWN') for item in data)

    print(f"\n{label}")
    print(f"Total examples: {total}")
    print(f"Case distribution:")
    for case_label in sorted(case_counts):
        count = case_counts[case_label]
        print(f" {case_label}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":
    main()