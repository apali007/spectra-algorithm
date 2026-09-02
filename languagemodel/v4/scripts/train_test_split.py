# ==============
# Main train /test split script for the prospect reasoning dataset.
import json
import random
from collections import defaultdict

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_reasoning_dataset.json', 'r') as f:
    dataset = json.load(f)

print(f"Total spectra in dataset: {len(dataset)}")

# dataset = [entry for entry in dataset if entry['is_label_match'] and entry['is_precusor_match']]

print(f"Total spectra after filtering for label and precursor matches: {len(dataset)}")

peptide_groups = defaultdict(list)
for entry in dataset:
    peptide_groups[entry['peptide']].append(entry)

unique_peptides = list(peptide_groups.keys())
random.seed(42)
random.shuffle(unique_peptides)

split_idx = int(len(unique_peptides) * 0.8)
train_peptides = set(unique_peptides[:split_idx])
test_peptides = set(unique_peptides[split_idx:])

train_set = [entry for p in train_peptides for entry in peptide_groups[p]]
test_set = [entry for p in test_peptides for entry in peptide_groups[p]]

print(f"Spectra - train: {len(train_set)}, test: {len(test_set)}")
print(f"Unique peptides - train: {len(train_peptides)}, test: {len(test_peptides)}")

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/all_clean_y_ion_case_prospect_reasoning_dataset_train.json', 'w') as f:
    json.dump(train_set, f, indent=2)

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/all_clean_y_ion_case_prospect_reasoning_dataset_test.json', 'w') as f:
    json.dump(test_set, f, indent=2)

print("\nDone.")

