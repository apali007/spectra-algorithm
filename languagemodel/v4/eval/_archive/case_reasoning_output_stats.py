import json
from collections import Counter, defaultdict

def detect_case(output):
    if "No need to derive a b-ion ladder as the complete sequence is confirmed." in output:
        return "CASE_CLEAN_Y_ION_LADDER"
    if "Resolving the remaining b-ion ladder gaps using the y-ion ladder:" not in output:
        return "CASE_CLEAN_B_ION_LADDER" if "The b-ion ladder identifies this sequence" in output else None
    if "Combining both ladders gives the full predicted sequence:" in output:
        return "CASE_COMPLEMENTARY_GAPS"
    if "Since exactly one residue remains unresolved, it can be recovered directly from the precursor mass:" in output:
        return "CASE_SINGLE_GAP_RESOLVED"
    if "The complete sequence cannot be predicted." in output:
        gaps = output.count("cannot be resolved by the y-ion ladder")
        return "CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH" if gaps == 1 else "CASE_UNRESOLVABLE_MULTI_GAP"
    return None


# dataset_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect-results-inference-spectra-full.json'  # set this
dataset_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-full.json'

print(dataset_path)

with open(dataset_path) as f:
    corpus = json.load(f)

total = len(corpus)
print(total)

total_by_case = Counter(e["case_label"] for e in corpus)

print(f"\nLabeled Case distribution:")
for case_label in sorted(total_by_case):

    count = total_by_case[case_label]
    print(f" {case_label}: {count} ({100*count/total:.1f}%)")


matches = [e for e in corpus if detect_case(e["prediction"] if e["prediction"] is not None else "CASE_NO_EXTRACTION") == e["case_label"]]
match_by_case = Counter(e["case_label"] for e in matches)

print(f"\nPredicted overall case matches: {len(matches)} / {total} ({100*len(matches)/total:.1f}%)")
for case, case_total in total_by_case.items():
    count = match_by_case.get(case, 0)
    print(f"  {case}: {count} / {case_total} ({100*count/case_total:.1f}%)")

# what did the model actually generate, for each expected case?
print("\nWhat the model actually produced, by expected case:")
generated_by_expected = defaultdict(Counter)
for e in corpus:
    generated_by_expected[e["case_label"]][detect_case(e["prediction"]) if e["prediction"] is not None else "CASE_NO_EXTRACTION"] += 1

for expected_case, case_total in total_by_case.items():
    print(f"\n  expected: {expected_case}  (n={case_total})")
    for generated_case, count in generated_by_expected[expected_case].most_common():
        label = generated_case if generated_case else "UNPARSEABLE"
        print(f"    -> generated {label}: {count} ({100*count/case_total:.1f}%)")