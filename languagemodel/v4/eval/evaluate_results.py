import json
import re
from collections import Counter

CASE_CLEAN_B_ION_LADDER = "CASE_CLEAN_B_ION_LADDER"
CASE_CLEAN_Y_ION_LADDER = "CASE_CLEAN_Y_ION_LADDER"
CASE_COMPLEMENTARY_GAPS = "CASE_COMPLEMENTARY_GAPS"
CASE_UNRESOLVABLE_MULTI_GAP = "CASE_UNRESOLVABLE_MULTI_GAP"
CASE_SINGLE_GAP_RESOLVED = "CASE_SINGLE_GAP_RESOLVED"
CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH = "CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH"

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

VALID_RESIDUES = set(AA_MASSES.keys())

def extract_predicted_peptide(prediction):
    pep_level_match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
    if pep_level_match:
        return pep_level_match.group(1)
    
    return None
    
def normalize(peptide):
    return peptide.replace('I', 'L').replace('N', 'D')

def has_neutral_loss(text):
    return "neutral loss of" in text

def extract_charge(input_text):
    pep_level_match = re.search(r"Precursor Charge: (\d+)", input_text)
    return int(pep_level_match.group(1)) if pep_level_match else None

def mass(aa):
    return AA_MASSES.get(aa, float('nan'))

def has_invalid_residues(predicted_peptide):
    return any(aa not in VALID_RESIDUES for aa in predicted_peptide)

def evaluate_aa_level_performance(aa_level_match, aa_level_ground_truth_total, aa_level_pred_total, ground_truth, predicted_peptide):
    ground_truth_pep_length = len(ground_truth)
    pred_pep_length = len(predicted_peptide)
    aa_level_ground_truth_total += ground_truth_pep_length
    aa_level_pred_total += pred_pep_length
    ground_truth_residues = list(ground_truth)
    predicted_residues = list(predicted_peptide)

    for i in range(ground_truth_pep_length):

        if i>= pred_pep_length or abs(mass(ground_truth_residues[i]) - mass(predicted_residues[i])) >= 0.1:
            continue

        prefix_sum_label = sum(mass(j) for j in ground_truth_residues[:i])
        suffix_sum_label = sum(mass(j) for j in ground_truth_residues[i:ground_truth_pep_length])
        prefix_sum_pred = sum(mass(j) for j in predicted_residues[:i])
        suffix_sum_pred = sum(mass(j) for j in predicted_residues[i:ground_truth_pep_length])

        prefix_match = abs(prefix_sum_label - prefix_sum_pred) <= 0.5
        suffix_match = abs(suffix_sum_label - suffix_sum_pred) <= 0.5
        aa_level_match += 1 if (prefix_match or suffix_match) else 0 

    return aa_level_match, aa_level_ground_truth_total, aa_level_pred_total
             
def evaluate(results_path):

    print(results_path)

    with open(results_path, 'r') as f:        
        results = json.load(f)

    total = len(results)
    print(f"\nTotal: {total}")

    unresolvable_total = 0
    pep_level_total = 0
    pep_level_match = 0
    aa_level_ground_truth_total = 0
    aa_level_match = 0
    aa_level_pred_total = 0
    no_extraction = 0
    invalid_residue = 0

    case_totals = Counter()
    case_matches = Counter()

    pep_len_totals = Counter()
    pep_len_matches = Counter()

    nl_totals = Counter()
    nl_matches = Counter()

    precursor_charge_totals = Counter()
    precursor_charge_matches = Counter()

    for item in results:
        case_label = item.get('case_label', 'UNKNOWN')
        case_totals[case_label] += 1

        if case_label in {CASE_UNRESOLVABLE_MULTI_GAP, CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH}:
            unresolvable_total += 1
            continue

        pep_level_total +=1
        ground_truth = item.get('ground_truth')
        prediction = item.get('prediction')
        pep_len = len(ground_truth)
        charge = extract_charge(item.get('input', ''))
        
        pep_len_totals[pep_len] += 1
        precursor_charge_totals[charge] += 1

        predicted_peptide = extract_predicted_peptide(prediction) if prediction is not None else None

        if predicted_peptide is None:
            no_extraction += 1 
            continue

        used_nl = has_neutral_loss(prediction)
        nl_totals[used_nl] += 1

        if has_invalid_residues(predicted_peptide):
            invalid_residue +=1

        aa_level_match, aa_level_ground_truth_total, aa_level_pred_total   = evaluate_aa_level_performance(aa_level_match, aa_level_ground_truth_total, aa_level_pred_total, ground_truth, predicted_peptide)

        if normalize(predicted_peptide) == normalize(ground_truth):
            pep_level_match +=1
            case_matches[case_label] += 1
            pep_len_matches[pep_len] += 1
            nl_matches[used_nl] += 1
            precursor_charge_matches[charge] += 1 


    pred_extracted_total = pep_level_total - no_extraction
    peptide_level_precision = pep_level_match/pred_extracted_total if pred_extracted_total else 0.0
    peptide_level_coverage = pred_extracted_total/pep_level_total
    peptide_level_total_pred_accuracy = pep_level_match/pep_level_total
    # aa_level_recall = aa_level_match/aa_level_ground_truth_total if aa_level_ground_truth_total else 0.0
    aa_level_precision = aa_level_match/aa_level_pred_total if aa_level_pred_total else 0.0

    print(f"\nLabeled Case distribution:")
    for case_label in sorted(case_totals):

        case_label_count = case_totals[case_label]
        print(f" {case_label}: {case_label_count} ({100*case_label_count/total:.1f}%)")

    print(f"\nUnresolvable instances ({CASE_UNRESOLVABLE_MULTI_GAP}, {CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH} are excluded in the results below): {unresolvable_total} ({100*unresolvable_total/total:.1f}%)")
    print(f"\n Total Resolvable instances: {pep_level_total}")
    print(f"No prediction extracted (regex miss): {no_extraction} ({100*no_extraction/pep_level_total:.1f}%)")
    print(f"No of peptides with invalid residue in prediction: {invalid_residue} ({100*invalid_residue/pep_level_total:.1f}%)")
    print(f"Peptide level Coverage : {100*peptide_level_coverage:.1f}%")
    print(f"Peptide level Precision : {100*peptide_level_precision:.1f}%")
    print(f"Peptide level total predicted accuracy: {100*peptide_level_total_pred_accuracy:.1f}%")

    # print(f"Amino acid level recall : {100*aa_level_recall:.1f}%")
    print(f"Amino acid level Precision : {100*aa_level_precision:.1f}%")


    print("\n matched case_label:")
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
    results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-full.json'
    # results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_results-inference-spectra-full.json'
    # results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect-results-inference-spectra-full.json'

    evaluate(results_path)