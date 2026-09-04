# ==============
# Takes the first N examples from a generated reasoning-corpus JSON.
#
# Useful for producing a smaller dataset without re-running reasoning_corpus.py
# (which pays the same fixed loading/filtering overhead regardless of
# NUM_EXAMPLES) — since spectra are processed in a fixed, deterministic
# order, slicing an already-generated full dataset is equivalent to what a
# fresh smaller-NUM_EXAMPLES run would have produced, just far faster.
#
# Usage:
#   python scripts/slice_dataset.py \
#       --input output/mass_analyzer_aware_prospect_reasoning_dataset.json \
#       --n 40000 \
#       --output output/40k_mass_analyzer_aware_prospect_reasoning_dataset.json

import argparse
import json
import random
from pathlib import Path
from collections import Counter

DEFAULT_INPUT = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/mass_analyzer_aware_prospect_reasoning_dataset.json'


def main():
    parser = argparse.ArgumentParser(description="Take N examples from a reasoning corpus JSON")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                         help="Path to the generated reasoning-corpus JSON to slice.")
    parser.add_argument("--n", type=int, required=True,
                         help="Number of examples to keep.")
    parser.add_argument("--output", required=True,
                         help="Path to write the sliced JSON to.")
    parser.add_argument("--random", action="store_true",
                         help="Randomly sample N examples instead of taking the first N. "
                              "Use this unless you specifically want positional slicing — "
                              "spectra are processed in file order by reasoning_corpus.py, "
                              "so a positional slice is NOT a representative sample of the "
                              "full corpus's case_label distribution (raw files are processed "
                              "in sequence, and case-type mix can vary systematically by file).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for --random sampling, for reproducibility.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as f:
        dataset = json.load(f)

    input_unique_peptides = len({item.get('peptide') for item in dataset})
    print(f"Input: {input_path}")
    print(f"Total spectra: {len(dataset)}")
    print(f"Unique peptides: {input_unique_peptides}")

    if args.n > len(dataset):
        print(f"WARNING: requested {args.n} but only {len(dataset)} available — writing all of them.")

    if args.random:
        random.seed(args.seed)
        subset = random.sample(dataset, min(args.n, len(dataset)))
        print(f"\nRandomly sampled {len(subset)} examples (seed={args.seed})")
    else:
        subset = dataset[:args.n]
        print(f"\nTook the first {len(subset)} examples (positional — see --random if this should "
              f"instead be a representative sample)")

    subset_unique_peptides = len({item.get('peptide') for item in subset})
    print(f"Spectra in slice: {len(subset)}")
    print(f"Unique peptides in slice: {subset_unique_peptides}")

    case_counts = Counter(item.get('case_label', 'UNKNOWN') for item in subset)
    print("Case distribution in slice:")
    for label in sorted(case_counts):
        count = case_counts[label]
        print(f" {label}: {count} ({100*count/len(subset):.1f}%)")

    with open(output_path, 'w') as f:
        json.dump(subset, f, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()