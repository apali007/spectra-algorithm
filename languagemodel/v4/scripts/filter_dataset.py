import json
import re

INPUT_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/input_data/TUM_third_pool/mass_analyzer_aware_prospect_reasoning_dataset.json"
OUTPUT_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/input_data/TUM_third_pool/clean_y_ion_FTMS_charge_2_no_nl_pep_len_12_dataset.json"
MAX_LENGTH = 12

with open(INPUT_PATH) as f:
    data = json.load(f)

def get_charge(text):
    m = re.search(r"Precursor Charge:\s*(\d+)", text)
    return int(m.group(1)) if m else None

kept = []
for item in data:
    if item.get("case_label") != "CASE_CLEAN_Y_ION_LADDER":
        continue
    if item.get("mass_analyzer") != "FTMS":
        continue
    charge = get_charge(item.get("input", ""))
    if charge is None or charge > 2:
        continue
    if "neutral loss" in item.get("output", "").lower():
        continue
    if len(item.get("peptide", "")) > MAX_LENGTH:
        continue
    kept.append(item)

print(f"{len(kept)} / {len(data)} kept ({(len(kept)/len(data))*100:.2f}%)")

with open(OUTPUT_PATH, "w") as f:
    json.dump(kept, f, indent=2)