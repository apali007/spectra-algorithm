# # ==============
# # Split the test set into two halves for two GPU inference runs.
import json

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/all_clean_y_ion_case_prospect_reasoning_dataset_test.json') as f:
#     test_set = json.load(f)

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/30k_all_cases_prospect_reasoning_dataset_test.json') as f:
    test_set = json.load(f)

mid = len(test_set) // 2
half1 = test_set[:mid]
half2 = test_set[mid:]

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/all_clean_y_ion_case_prospect_reasoning_dataset_half1.json', 'w') as f:
#     json.dump(half1, f, indent=2)

# with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/all_clean_y_ion_case_prospect_reasoning_dataset_half2.json', 'w') as f:
#     json.dump(half2, f, indent=2)

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/30k_all_cases_prospect_reasoning_dataset_half1.json', 'w') as f:
    json.dump(half1, f, indent=2)

with open('/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/30k_all_cases_prospect_reasoning_dataset_half2.json', 'w') as f:
    json.dump(half2, f, indent=2)


print(f"Half 1 size: {len(half1)}, Half 2 size: {len(half2)}")
