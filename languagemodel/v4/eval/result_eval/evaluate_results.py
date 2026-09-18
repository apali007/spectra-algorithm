import json
import re
from collections import Counter, defaultdict
import os
import sys
import argparse
from casanovo_eval import aa_match_batch, aa_match_metrics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from constants import (
    AA_MASSES,
    CASE_CLEAN_B_ION_LADDER,
    CASE_CLEAN_Y_ION_LADDER,
    CASE_COMPLEMENTARY_GAPS,
    CASE_UNRESOLVABLE_MULTI_GAP,
    CASE_SINGLE_GAP_RESOLVED,
    CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH,
)

VALID_RESIDUES = set(AA_MASSES.keys())



def detect_case(output):
    if "No need to derive a b-ion ladder as the complete sequence is confirmed." in output:
        return CASE_CLEAN_Y_ION_LADDER
    if "Resolving the remaining b-ion ladder gaps using the y-ion ladder:" not in output:
        return CASE_CLEAN_B_ION_LADDER if "The b-ion ladder identifies this sequence" in output else None
    if "Combining both ladders gives the full predicted sequence:" in output:
        return CASE_COMPLEMENTARY_GAPS
    if "Since exactly one residue remains unresolved, it can be recovered directly from the precursor mass:" in output:
        return CASE_SINGLE_GAP_RESOLVED
    if "The complete sequence cannot be predicted." in output:
        gaps = output.count("cannot be resolved by the y-ion ladder")
        return CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH if gaps == 1 else CASE_UNRESOLVABLE_MULTI_GAP
    return None


def extract_predicted_peptide(prediction):
    pep_level_match = re.search(r"Final predicted sequence: ([A-Z]+)\.", prediction)
    if pep_level_match:
        return pep_level_match.group(1)
    
    return None
    
def normalize(peptide):
    return peptide.replace('I', 'L')

def has_neutral_loss(text):
    return "neutral loss of" in text

def extract_charge(input_text):
    pep_level_match = re.search(r"Precursor Charge: (\d+)", input_text)
    return int(pep_level_match.group(1)) if pep_level_match else None


def extract_mass_analyzer(input_text: str) -> str | None:
    match = re.search(r"Mass Analyzer:\s*(FTMS|ITMS)", input_text)
    return match.group(1) if match else None

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
             
def evaluate(results_path, output_path, mode):

    print(results_path)

    with open(results_path, 'r') as f:        
        results = json.load(f)

    total = len(results)
    print(f"\nTotal: {total}")

    if total == 0:
        print("Empty result set - nothing to evaluate.")
        return

    unresolvable_total = 0
    resolvable_total = 0
    pep_level_match = 0
    aa_level_ground_truth_total = 0
    aa_level_match = 0
    aa_level_pred_total = 0
    no_extraction = 0
    invalid_residue = 0

    case_totals = Counter()
    case_matches = Counter()

    case_no_extraction = Counter()

    pep_len_totals = Counter()
    pep_len_matches = Counter()

    nl_totals = Counter()
    nl_matches = Counter()

    precursor_charge_totals = Counter()
    precursor_charge_matches = Counter()

    mass_analyzer_totals = Counter()
    mass_analyzer_matches = Counter()

    reasoning_case_matches = Counter()
    generated_by_expected = defaultdict(Counter)

    pred_seqs, true_seqs = [], []

    for item in results:
        if mode == "output":
            ground_truth = item.get('ground_truth')
            prediction = item.get('prediction')
        else:
            ground_truth = item.get('peptide')
            prediction = item.get('output')

        case_label = item.get('case_label', 'UNKNOWN')
        case_totals[case_label] += 1

        detected_case = detect_case(prediction) if prediction is not None else "CASE_NO_EXTRACTION"
        generated_by_expected[case_label][detected_case if detected_case else "UNPARSEABLE"] += 1
        if detected_case == case_label:
            reasoning_case_matches[case_label] += 1

        if case_label in {CASE_UNRESOLVABLE_MULTI_GAP, CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH}:
            unresolvable_total += 1
            continue

        # RESOLABLE CASES
        resolvable_total +=1

        pep_len = len(ground_truth)
        charge = extract_charge(item.get('input', ''))
        mass_analyzer = extract_mass_analyzer(item.get('input', ''))
        
        pep_len_totals[pep_len] += 1
        precursor_charge_totals[charge] += 1
        mass_analyzer_totals[mass_analyzer] += 1

        predicted_peptide = extract_predicted_peptide(prediction) if prediction is not None else None

        pred_seqs.append(predicted_peptide)
        true_seqs.append(ground_truth)

        if predicted_peptide is None:
            no_extraction += 1 
            case_no_extraction[case_label] += 1
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
            mass_analyzer_matches[mass_analyzer] +=1


    pred_extracted_total = resolvable_total - no_extraction
    peptide_level_precision = pep_level_match/pred_extracted_total if pred_extracted_total else 0.0
    peptide_level_coverage = pred_extracted_total/resolvable_total if resolvable_total else 0.0
    peptide_level_total_pred_accuracy = pep_level_match/resolvable_total if resolvable_total else 0.0
    # aa_level_recall = aa_level_match/aa_level_ground_truth_total if aa_level_ground_truth_total else 0.0
    aa_level_precision = aa_level_match/aa_level_pred_total if aa_level_pred_total else 0.0

    print(f"\nLabeled Case distribution:")
    for case_label in sorted(case_totals):

        case_label_count = case_totals[case_label]
        case_label_count_pct = 100*case_label_count/total if total else 0.0
        print(f" {case_label}: {case_label_count} ({case_label_count_pct:.1f}%)")

    unresolvable_total_pct = 100*unresolvable_total/total if total else 0.0
    print(f"\nUnresolvable instances ({CASE_UNRESOLVABLE_MULTI_GAP}, {CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH} are excluded in the results below): {unresolvable_total} ({unresolvable_total_pct:.1f}%)")
    resolvable_total_pct = 100*resolvable_total/total if total else 0.0
    print(f"\n Total Resolvable instances: {resolvable_total} ({resolvable_total_pct:.1f}%)")

    no_extraction_pct = 100*no_extraction/resolvable_total if resolvable_total else 0.0
    print(f"\nNo prediction extracted (regex miss): {no_extraction} ({no_extraction_pct:.1f}%)")

    if case_no_extraction:
        print(f"\nNo extraction Case distribution:")
        for case_label in sorted(case_no_extraction):

            case_label_count = case_no_extraction[case_label]
            case_label_count_pct = 100*case_label_count/resolvable_total if resolvable_total else 0.0
            print(f" {case_label}: {case_label_count} ({case_label_count_pct:.1f}%)")

    invalid_residue_pct = 100*invalid_residue/resolvable_total if resolvable_total else 0.0
    print(f"\nNo of peptides with invalid residue in prediction: {invalid_residue} ({invalid_residue_pct:.1f}%)")
    print(f"Peptide level Coverage : {pred_extracted_total} ({100*peptide_level_coverage:.1f}%) ")
    print(f"Peptide level Precision : {100*peptide_level_precision:.1f}%")
    print(f"Peptide level total predicted accuracy: {100*peptide_level_total_pred_accuracy:.1f}%")

    # print(f"Amino acid level recall : {100*aa_level_recall:.1f}%")
    print(f"Amino acid level Precision : {100*aa_level_precision:.1f}%")


    print("\n performance by case_label:")
    for case_label in sorted(case_totals):
        c_total = case_totals[case_label]
        c_match = case_matches[case_label]
        print(f" {case_label}: {c_match}/{c_total} ({100*c_match/c_total:.1f}%)")

    print("\n performance by peptide length:")
    for pep_length in sorted(pep_len_totals):
        p_total = pep_len_totals[pep_length]
        p_match = pep_len_matches[pep_length]
        print(f" peptide length {pep_length}: {p_match}/{p_total} ({100*p_match/p_total:.1f}%)")

    print("\n performance by neutral loss:")
    for used_nl in [True, False]:
        n_total = nl_totals[used_nl]
        n_match = nl_matches[used_nl]
        label = "used neutral loss" if used_nl else "no neutral loss"
        if n_total:
            print(f" {label}: {n_match}/{n_total} ({100*n_match/n_total:.1f}%)")

    print("\n performance by precursor charge:")
    for charge in sorted(precursor_charge_totals, key=lambda x: (x is None, x)):
        ch_total = precursor_charge_totals[charge]
        ch_match = precursor_charge_matches[charge]
        print(f" precursor charge {charge}: {ch_match}/{ch_total} ({100*ch_match/ch_total:.1f}%)")

    print("\n performance by mass analyzer:")
    for mass_analyzer in sorted(mass_analyzer_totals, key=lambda x: (x is None, x)):
        ma_total = mass_analyzer_totals[mass_analyzer]
        ma_match = mass_analyzer_matches[mass_analyzer]
        print(f" mass_analyzer {mass_analyzer}: {ma_match}/{ma_total} ({100*ma_match/ma_total:.1f}%)")

    print("\n reasoning-structure match (did it follow the right derivation pattern?):")
    total_reasoning_matches = sum(reasoning_case_matches.values())
    reasoning_match_pct = 100*total_reasoning_matches/total if total else 0.0
    print(f" Overall: {total_reasoning_matches}/{total} ({reasoning_match_pct:.1f}%)")

    for case_label in sorted(case_totals):
        c_total = case_totals[case_label]
        r_match = reasoning_case_matches[case_label]
        pct = 100*r_match/c_total if c_total else 0.0
        print(f" {case_label}: {r_match}/{c_total} ({pct:.1f}%)")

    print(" \n what the model actually produced, by expected case:")
    for expected_case in sorted(case_totals):
        c_total = case_totals[expected_case]
        print(f"\n expected: {expected_case} (n={c_total})")
        for generated_case, count in generated_by_expected[expected_case].most_common():
            label = generated_case if generated_case else "UNPARSEABLE"
            pct = 100*count/c_total if c_total else 0.0
            print(f" -> generated {label}: {count} ({pct:.1f}%)")


    print(" \n Casanovo evaluation metric: ")
    casanovo_aa_precision, casanovo_aa_recall, casanovo_pep_precision = aa_match_metrics(*aa_match_batch(true_seqs, pred_seqs, AA_MASSES))
    print(f"casanovo_aa_precision: {casanovo_aa_precision*100:.2f}")
    print(f"casanovo_aa_recall: {casanovo_aa_recall*100:.2f}")
    print(f"casanovo_pep_precision: {casanovo_pep_precision*100:.2f}")

    metrics = {
        "input_path": results_path,
        "output_path": output_path,
        "mode": mode,
        "total": total,
        "case_distribution": dict(case_totals),
        "unresolvable_total": unresolvable_total,
        "resolvable_total": resolvable_total,
        "no_extraction": no_extraction,
        "no_extraction_by_case": dict(case_no_extraction),
        "invalid_residue": invalid_residue,
        "peptide_level_coverage_pct": 100*peptide_level_coverage,
        "peptide_level_precision_pct": 100*peptide_level_precision,
        "peptide_level_total_accuracy_pct": 100*peptide_level_total_pred_accuracy,
        "aa_level_precision_pct": 100*aa_level_precision,
        "accuracy_by_case_label": {
            case_label: {
                "total": case_totals[case_label],
                "matches": case_matches[case_label],
                "accuracy_pct": 100*case_matches[case_label]/case_totals[case_label] if case_totals[case_label] else 0.0,
            }
            for case_label in sorted(case_totals)
        },
        "accuracy_by_peptide_length": {
            pep_length: {
                "total": pep_len_totals[pep_length],
                "matches": pep_len_matches[pep_length],
                "accuracy_pct": 100*pep_len_matches[pep_length]/pep_len_totals[pep_length] if pep_len_totals[pep_length] else 0.0,
            }
            for pep_length in sorted(pep_len_totals)
        },
        "accuracy_by_neutral_loss": {
            str(used_nl): {
                "total": nl_totals[used_nl],
                "matches": nl_matches[used_nl],
                "accuracy_pct": 100*nl_matches[used_nl]/nl_totals[used_nl] if nl_totals[used_nl] else 0.0,
            }
            for used_nl in [True, False] if nl_totals[used_nl]
        },
        "accuracy_by_precursor_charge": {
            str(charge): {
                "total": precursor_charge_totals[charge],
                "matches": precursor_charge_matches[charge],
                "accuracy_pct": 100*precursor_charge_matches[charge]/precursor_charge_totals[charge] if precursor_charge_totals[charge] else 0.0,
            }
            for charge in sorted(precursor_charge_totals, key=lambda x: (x is None, x))
        },
        "accuracy_by_mass_analyzer": {
                    str(mass_analyzer): {
                        "total": mass_analyzer_totals[mass_analyzer],
                        "matches": mass_analyzer_matches[mass_analyzer],
                        "accuracy_pct": 100*mass_analyzer_matches[mass_analyzer]/mass_analyzer_totals[mass_analyzer] if mass_analyzer_totals[mass_analyzer] else 0.0,
                    }
                    for mass_analyzer in sorted(mass_analyzer_totals, key=lambda x: (x is None, x))
                },
        "reasoning_structure_match": {
            "overall_pct": 100*sum(reasoning_case_matches.values())/total if total else 0.0,
            "by_case_label": {
                case_label: {
                    "total": case_totals[case_label],
                    "matches": reasoning_case_matches[case_label],
                    "match_pct": 100*reasoning_case_matches[case_label]/case_totals[case_label] if case_totals[case_label] else 0.0,
                }
                for case_label in sorted(case_totals)
            },
        },
        "generated_case_breakdown": {
            expected_case: dict(generated_by_expected[expected_case])
            for expected_case in sorted(case_totals)
        },
        "casanovo_evaluation_metric": {
            "casanovo_aa_precision": casanovo_aa_precision*100,
            "casanovo_aa_recall": casanovo_aa_recall*100,
            "casanovo_pep_precision": casanovo_pep_precision*100,
        },
    }

    if output_path is not None:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved evaluation metrics to {output_path}")



    

def build_argparser():
    # results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/30k_all_cases_prospect_results-inference-spectra-full.json'
    # results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/all_clean_y_ion_case_prospect_results-inference-spectra-full.json'
    # results_path = '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect-results-inference-spectra-full.json'

    parser = argparse.ArgumentParser(description="Evaluate predictions from either model output or the reasoning corpus")
    parser.add_argument("--input", type=str, required=True, help="Path to the results JSON")
    parser.add_argument("--mode", choices=["output", "corpus"], required=True,
                         help="'output' reads inference.py's prediction/ground_truth fields; "
                              "'corpus' reads the raw dataset's output/peptide fields as a "
                              "data-quality check (does the corpus's own derivation match its label)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save evaluation results as JSON. "
                             "If not provided, results are only printed to stdout.")
                              
    return parser

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    evaluate(args.input, args.output, args.mode)