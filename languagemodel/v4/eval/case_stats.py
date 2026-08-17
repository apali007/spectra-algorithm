import json 
from collections import Counter

def show_case_stats(dataset_path):

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    total = len(data)
    case_counts = Counter(item.get('case_label', 'UNKNOWN') for item in data)

    print(f"\nDataset: {dataset_path}")
    print(f"Total examples: {total}")
    print(f"\nCase distribution:")
    for case_label in sorted(case_counts):

        count = case_counts[case_label]
        print(f" {case_label}: {count} ({100*count/total:.1f}%)")


if __name__ == "__main__":

    # dataset_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect-results-inference-spectra-full.json'
    dataset_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect_reasoning_dataset_v1.json'

    show_case_stats(dataset_path)