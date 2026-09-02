import json
from collections import Counter

train_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/prospect_reasoning_dataset_train.json'
test_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/prospect_reasoning_dataset_test.json'

with open(train_path) as f:
    train = json.load(f)
with open(test_path) as f:
    test = json.load(f)

def has_neutral_loss(text):
    return "neutral loss of" in text

train_nl = Counter(has_neutral_loss(e['output']) for e in train)
test_nl = Counter(has_neutral_loss(e['output']) for e in test)

train_total = len(train)
test_total = len(test)

print("Neutral loss distribution — training set")
for used_nl in [True, False]:
    count = train_nl.get(used_nl, 0)
    label = "used neutral loss" if used_nl else "no neutral loss"
    print(f"  {label}: {count} ({100*count/train_total:.1f}%)")

print("\nNeutral loss distribution — test set")
for used_nl in [True, False]:
    count = test_nl.get(used_nl, 0)
    label = "used neutral loss" if used_nl else "no neutral loss"
    print(f"  {label}: {count} ({100*count/test_total:.1f}%)")