import json
from collections import Counter

train_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/prospect_reasoning_dataset_train.json'
test_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/prospect_reasoning_dataset_test.json'

with open(train_path) as f:
    train = json.load(f)

with open(test_path) as f:
    test = json.load(f)

train_len_counts = Counter(len(e['peptide']) for e in train)
test_len_counts = Counter(len(e['peptide']) for e in test)

train_total = len(train)
test_total = len(test)

print("Peptide length distribution - training set")
for length in sorted(train_len_counts):
    count = train_len_counts[length]
    print(f" peptide len {length}: {count} ({100*count/train_total:.1f}%)")

print("\nPeptide length distribution - test set")
for length in sorted(test_len_counts):
    count = test_len_counts[length]
    print(f" peptide len {length}: {count} ({100*count/test_total:.1f}&)")