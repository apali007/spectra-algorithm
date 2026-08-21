import json
import re
from collections import Counter

def extract_charge(input_text):
    match = re.search(r"Precursor Charge: (\d+)", input_text)
    return int(match.group(1)) if match else None

train_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json'
test_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json'

with open(train_path) as f:
    train = json.load(f)
with open(test_path) as f:
    test = json.load(f)

train_charge = Counter(extract_charge(e['input']) for e in train)
test_charge = Counter(extract_charge(e['input']) for e in test)

train_total = len(train)
test_total = len(test)

print("Precursor charge distribution — training set")
for charge in sorted(train_charge):
    count = train_charge[charge]
    print(f"  charge {charge}: {count} ({100*count/train_total:.1f}%)")

print("\nPrecursor charge distribution — test set")
for charge in sorted(test_charge):
    count = test_charge[charge]
    print(f"  charge {charge}: {count} ({100*count/test_total:.1f}%)")