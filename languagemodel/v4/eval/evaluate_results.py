import json
import re
from collections import Counter

def extract_predicted_peptide(prediction):
    match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
    if match:
        return match.group(1)
    
    return None
    
def normalize(peptide):
    return peptide.replace('I', 'L').replace('N', 'D')

def has_neutral_loss(text):
    return "neutral loss of" in text

def extract_charge(input_text):
    match = re.search(r"Precursor Charge: (\d+)", input_text)
    return int(match.group(1)) if match else None

def evaluate(results_path):

    with open(results_path, 'r') as f:        
        results = json.load(f)

    total = 0
    match = 0
    no_extraction = 0

    case_totals = Counter()
    case_matches = Counter()

    pep_len_totals = Counter()
    pep_len_matches = Counter()

    nl_totals = Counter()
    nl_matches = Counter()

    precursor_charge_totals = Counter()
    precursor_charge_matches = Counter()

    for item in results:
        total +=1
        ground_truth = item.get('ground_truth')
        prediction = item.get('prediction')
        case_label = item.get('case_label', 'UNKNOWN')
        pep_len = len(ground_truth)
        charge = extract_charge(item.get('input', ''))
        used_nl = has_neutral_loss(prediction)
                

        case_totals[case_label] += 1
        pep_len_totals[pep_len] += 1
        nl_totals[used_nl] += 1
        precursor_charge_totals[charge] += 1

        predicted_peptide = extract_predicted_peptide(prediction)

        if predicted_peptide is None:
            no_extraction += 1 
            continue

        if normalize(predicted_peptide) == normalize(ground_truth):
            match +=1
            case_matches[case_label] += 1
            pep_len_matches[pep_len] += 1
            nl_matches[used_nl] += 1
            precursor_charge_matches[charge] += 1 

    print(f"\n Total: {total}")
    print(f"No prediction extracted (regex miss): {no_extraction} ({100*no_extraction/total:.1f}%)")
    print(f"Accuracy : {100*match/total:.1f}%")

    print("\n case_label:")
    for case_label in sorted(case_totals):
        c_total = case_totals[case_label]
        c_match = case_matches[case_label]
        print(f" {case_label}: {c_match}/{c_total} ({100*c_match/c_total:.1f}%)")

    print("\n peptide length:")
    for pep_length in sorted(pep_len_totals):
        p_total = pep_len_totals[pep_length]
        p_match = pep_len_matches[pep_length]
        print(f" peptide length {pep_length}: {p_match}/{p_total} ({100*p_match/p_total:.1f}%)")

    print("\n neutral loss:")
    for used_nl in [True, False]:
        n_total = nl_totals[used_nl]
        n_match = nl_matches[used_nl]
        label = "used neutral loss" if used_nl else "no neutral loss"
        if n_total:
            print(f" {label}: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")

    print("\n precursor charge:")
    for charge in sorted(precursor_charge_totals, key=lambda x: (x is None, x)):
        ch_total = precursor_charge_totals[charge]
        ch_match = precursor_charge_matches[charge]
        print(f" precursor charge {charge}: {ch_match}/{ch_total} ({100*ch_match/ch_total:.1f}%)")

if __name__ == "__main__":

    results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_results-inference-spectra-full.json'

    evaluate(results_path)