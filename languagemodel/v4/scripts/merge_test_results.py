# # ==============
# # Merge the test results from two halves of two GPU inference runs to one.
import json

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-half1.json') as f:
    results1 = json.load(f)

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-half2.json') as f:
    results2 = json.load(f)

merged = results1 + results2
print(f"Total merged: {len(merged)}")

for new_id, entry in enumerate(merged, 1):
    entry['spectra_id'] = new_id

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-full.json', 'w') as f:
    json.dump(merged, f, indent =2)