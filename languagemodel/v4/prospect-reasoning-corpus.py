import json
import re
import numpy as np
import pandas as pd
import glob
import os
import pickle

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

NEUTRAL_LOSS_MASSES = {
    'H2O': 18.01056,
    'NH3': 17.02655
}

# MASS_H2O    = 18.01056
# MASS_NH3 = 17.02655
MASS_PROTON =  1.00728
MASS_C13    = 1.00335 # Mass Difference between C-12 and C-13

def validate_precursor(merged_sequence, precursor_mz, precursor_charge, precursor_tolerance):

    if any(aa is None for aa in merged_sequence):
        return False, None

    theoretical_mass = sum(AA_MASSES[r] for r in merged_sequence) +  NEUTRAL_LOSS_MASSES['H2O']
    precursor_mass_neutral = precursor_mz * precursor_charge - precursor_charge * MASS_PROTON
    diff = abs(precursor_mass_neutral - theoretical_mass)

    if diff < precursor_tolerance:
        return True, diff
    else:
        return False, diff


def merge_y_b_ions(y_gap_filled, b_gap_filled, n):
    merged_sequence = [None] * n

    y_gap_filled_reversed = list(reversed(list(y_gap_filled)))
    for i in range(n):
        if y_gap_filled_reversed[i] is not None:
            merged_sequence[i] = y_gap_filled_reversed[i]
        elif b_gap_filled[i] is not None:
             merged_sequence[i] = b_gap_filled[i]

    return merged_sequence

def handle_ion_gaps(ion_type, own_derived, other_derived, label_peptide, n):

    label_peptide = list(reversed(list(label_peptide))) if ion_type == 'y' else list(label_peptide)

    # n+1 is not needed because yn and bn is equal to precursor
    for i in range(n):
        if own_derived[i] is None:

            if other_derived[n-i-1] is not None:
                own_derived[i] = label_peptide[i]
                if VERBOSE:
                    print(f"{ion_type}{i+1} is filled by {'b' if ion_type == 'y' else 'y'}{n-i}")
                continue

            if i == 0 and other_derived[n-2] is not None:
                own_derived[i] = label_peptide[i]
                if VERBOSE:
                    print(f"{ion_type}{i+1} is filled by {'b' if ion_type == 'y' else 'y'}{n-1}")

    return own_derived

def is_mass_verified(i , ion_type, mass , ladder_residues, tolerance):
    theoretical_mass = sum(AA_MASSES[r] for r in ladder_residues[:i])
    
    if ion_type == 'y':
        theoretical_mass += NEUTRAL_LOSS_MASSES['H2O'] + MASS_PROTON
    else:
        theoretical_mass += MASS_PROTON

    difference = abs(mass - theoretical_mass)
    is_valid = difference < tolerance

    if VERBOSE:
        print(f"{ion_type}{i}, diff: {difference} | is_valid: {is_valid}")

    return is_valid

def handle_charge_nl(ion_type, mz, charge, nl, i):

    # multiple charge with nl
    if charge > 1 and nl is not None:
        singly_charged = mz * charge - (charge - 1) * MASS_PROTON
        correct_mass = singly_charged + NEUTRAL_LOSS_MASSES[nl]
       
        traces = (
                f"{ion_type}{i} can be identified by mz_peak at {mz:.5f} Da with charge {charge} "
                f"and neutral loss of {nl}. Converting into singly charged ion, it becomes "
                f"{singly_charged:.5f} Da. Correcting the mass, it becomes {correct_mass:.5f} Da."
                )
        final_mass = correct_mass

    # multiple charge with no nl
    elif charge > 1:
        singly_charged = mz * charge - (charge - 1) * MASS_PROTON
        traces = (
                f"{ion_type}{i} can be identified by mz_peak at {mz:.5f} Da with charge {charge}. "
                f"Converting into singly charged ion, it becomes {singly_charged:.5f} Da.")
        final_mass = singly_charged

    # single charge with nl
    elif nl is not None: 
        correct_mass = mz + NEUTRAL_LOSS_MASSES[nl]
        traces = (
                f"{ion_type}{i} can be identified by mz_peak at {mz:.5f} Da  with charge {charge} and neutral loss of {nl} Da. "
                f"Correcting the mass, it becomes {correct_mass:.5f} Da.")
        final_mass = correct_mass

    # single charge with no nl
    else:
        traces = f"{ion_type}{i} can be identified by mz_peak at {mz:.5f} Da. "
        final_mass = mz
    
    return traces, final_mass

def handle_first_position(mz, ion_type, ladder_residues, tolerance):
    if ion_type == 'y':
        predicted_aa_mass = mz-NEUTRAL_LOSS_MASSES['H2O']-MASS_PROTON
        trace = (
                f"Subtracting H2O ({NEUTRAL_LOSS_MASSES['H2O']:.5f} Da) and a proton ({MASS_PROTON:.5f} Da) "
                f"gives {predicted_aa_mass:.5f} Da. "
                f"Identifying the amino acid residue as {ladder_residues[0]}."
            )
    else:
        predicted_aa_mass = mz-MASS_PROTON
        trace = (
        f"Subtracting a proton ({MASS_PROTON:.5f} Da) gives {mz-MASS_PROTON:.5f} Da. "
        f"Identifying the amino acid residue as {ladder_residues[0]}."
        )
    
    return trace

def extend_ladder(ion_type, annotation_lookup, label_peptide, precursor_mz, precursor_charge, tolerance):
    residues = list(label_peptide)
    n = len(residues)
    ladder_residues = list(reversed(residues)) if ion_type == 'y' else residues

    traces = [f"Analyzing spectra for the {ion_type}-ion ladder: "]
    # Initializing the position of fragment ions to be derived with none.
    derived_aa_pos = [None] * n
    prev_mz = None

    for i in range(1, n + 1):
        peak_information = get_peak(annotation_lookup, ion_type, i, precursor_charge)
        if VERBOSE:
                print(f"{ion_type}{i}, peak_info: {peak_information}")
        
        if peak_information is None:
            prev_mz = None
            continue

        mz, charge, nl = peak_information
        trace_charge_nl, mass = handle_charge_nl(ion_type, mz, charge, nl, i)

        if not is_mass_verified(i, ion_type, mass, ladder_residues, tolerance):
            prev_mz = None
            continue

        traces.append(trace_charge_nl)          
        
        if i == 1: 
            traces.append(handle_first_position(mass, ion_type, ladder_residues, tolerance))
        elif prev_mz is not None:
            predicted_aa_mass = mass-prev_mz
            traces.append(
                    f"Difference between {mass:.5f} and {prev_mz:.5f} gives {predicted_aa_mass:.5f}, "
                    f"Identifying the amino acid residue as {ladder_residues[i-1]}. "
                )   
        else:
            predicted_aa_mass = mass
            traces.append(
                f"Identifying the amino acid residue as {ladder_residues[i-1]}. "
            )

        derived_aa_pos[i-1] = ladder_residues[i-1]
        prev_mz = mass

    detected_positions = [f"{ion_type}{i+1}" for i, aa in enumerate(derived_aa_pos) if aa is not None]
    gap_positions = [f"{ion_type}{i+1}" for i, aa in enumerate(derived_aa_pos) if aa is None]

    if VERBOSE:
        print(f"detected_positions: {', '.join(detected_positions)}")
        print(f"gap_positions: {', '.join(gap_positions)}")

    if gap_positions:
        traces.append(
            f"The {ion_type}-ion ladder confirmed residues at positions "
            f"{', '.join(detected_positions)}. No reliable fragment ion was found at "
            f"{', '.join(gap_positions)}. "
        )
    else:
        traces.append(
            f"The {ion_type}-ion ladder confirmed residues at every position: "
            f"{', '.join(detected_positions)}."
        )
    
    return derived_aa_pos, traces

def derive_peptide(annotation_rows, label_peptide, precursor_mz, precursor_charge, tolerance, precursor_tolerance):

    n = len(label_peptide)
    if VERBOSE:
        print(f"Length of the peptide: {n}")
        print(f"precursor_mz: {precursor_mz}")
        print(f"precursor_mz: {precursor_charge}")
    reasoning_trace = []
    annotation_lookup = build_annotation_lookup(annotation_rows)

    y_derived, y_traces = extend_ladder('y', annotation_lookup, label_peptide, precursor_mz, precursor_charge, tolerance)
    reasoning_trace.append('\n'.join(y_traces))

    reasoning_trace.append("Now, trying to derive the sequence using the b-ion ladder. ")

    b_derived, b_traces = extend_ladder('b', annotation_lookup, label_peptide, precursor_mz, precursor_charge, tolerance) 
    reasoning_trace.append('\n'.join(b_traces))

    y_gap_filled = handle_ion_gaps('y', y_derived, b_derived, label_peptide, n)
    b_gap_filled = handle_ion_gaps('b', b_derived, y_derived, label_peptide, n)
    merged_sequence = merge_y_b_ions(y_gap_filled, b_gap_filled, n)

    merged_str = ''.join(aa if aa is not None else '?' for aa in merged_sequence)

    is_precursor_match, precursor_diff = validate_precursor(merged_sequence, precursor_mz, precursor_charge, precursor_tolerance)
    is_match = label_peptide == merged_str
    
    if VERBOSE:
            print(f"merged_sequence: {merged_str}")
            print(f"is_precursor_match: {is_precursor_match}, diff: {precursor_diff}")

    complete_resoning_trace = '\n'.join(reasoning_trace)

    return is_match, is_precursor_match, merged_sequence, complete_resoning_trace

    
def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def spectrum_to_input_text(mz_list, intensity_list, precursor_mz, precursor_charge):
    unique_peaks = sorted(set(zip(mz_list, intensity_list)), key=lambda p: p[0])
    # peaks = sorted(zip(mz_list, intensity_list), key=lambda p: p[0])
    lines = [f"Precursor mz: {precursor_mz:.5f} Da, Precursor Charge: {precursor_charge}"]
    lines += ["Experimental MS/MS spectrum (m/z, intensity):"]
    lines += [f"m/z {mz:.5f}, intensity {intensity:.5f}" for mz, intensity in unique_peaks]
    lines.append("Analyze this spectrum and identify the most likely peptide sequence.")
    return "\n".join(lines)

def parse_mgf(mgf_path, max_valid_spectra=1000):
    spectra, current, valid_count = [], None, 0
    with open(mgf_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "BEGIN IONS":
                current = {"title": "", "peptide": None, "charge": 1, "pepmass": 0.0, "mz": [], "intensity": []}
            elif line == "END IONS":
                if current is not None:
                    pep = current.get("peptide", "")
                    if is_valid_peptide(pep):
                        spectra.append(current)
                        valid_count +=1
                        if valid_count >= max_valid_spectra:
                            break
                current = None
            elif current is None:
                continue
            elif line.startswith("TITLE="):
                current["title"] = line[6:]
            elif line.startswith("SEQ="):
                current["peptide"] = line[4:]
            elif line.startswith("PEPMASS="):
                current["pepmass"] = float(line.split("=",1)[1].split()[0])
            elif line.startswith("CHARGE="):
                try:
                    current["charge"] = int(line.split("=",1)[1].strip().rstrip("+="))
                except ValueError:
                    current["charge"] = 1
            elif line[0].isdigit() or line[0] == ".":
                parts = line.split()
                if len(parts) >= 2:
                    current["mz"].append(float(parts[0]))
                    current["intensity"].append(float(parts[1]))
    return spectra

def is_valid_peptide(peptide):
    return bool(peptide) and bool(re.fullmatch(r'[A-Z]+', peptide)) and all(aa in AA_MASSES for aa in peptide)

def build_peak_priority(precursor_charge):
    priority = []
    for nl in [None, 'H2O', 'NH3']:
        for charge in range(1, precursor_charge + 1):
            priority.append((charge, nl))
    return priority


def get_peak(annotation_lookup, ion_type, position, precursor_charge):
    
    for charge, nl in build_peak_priority(precursor_charge):
        k = (ion_type, position, charge, nl)
        if k in annotation_lookup:
            return annotation_lookup[k], charge, nl
    return None


def build_annotation_lookup(annotation_rows):
    
    annotation_lookup = {}
    for _, row in annotation_rows.iterrows():
        nl = row['neutral_loss'] if pd.notna(row['neutral_loss']) and row['neutral_loss'] else None
        key = (row['ion_type'], int(row['no']), int(row['charge']), nl)
        annotation_lookup[key] = float(row['experimental_mass'])
    
    return annotation_lookup

def load_prospect(annotation_data_path, meta_data_path, cache_path, max_spectra=1):

    # check if cache directory exists (100_prospect_cache.pkl)
    if os.path.exists(cache_path):
        if VERBOSE:
            print(f"Found cache at {cache_path}, loading instead of reprocessing")
        with open(cache_path, 'rb') as f:
            spectra_100 = pickle.load(f)
        return spectra_100[:max_spectra]


    meta_data = pd.read_parquet(meta_data_path)
    annotation_files = sorted(glob.glob(f"{annotation_data_path}/TUM_third_pool_*_01_01_annotation.parquet"))
    if VERBOSE:
        print(f"Found {len(annotation_files)} annotation files: {annotation_files}")

    annot_data = pd.concat(
        [pd.read_parquet(f) for f in annotation_files],
        ignore_index = True
    )

    if VERBOSE:
        print(f"initial_meta_data: number of PSM rows: {len(meta_data)}")
        print(f"initial_meta_data: total unique spectra in raw meta data file: {len(meta_data.groupby(['scan_number', 'raw_file']))}")
        print(f"initial_meta_data: total unique peptides in raw meta data file: {len(meta_data.groupby(['modified_sequence']))}")
        print(f"\nintial_annotated_data: total unique spectra in raw annotation file: {len(annot_data.groupby(['scan_number', 'raw_file']))}")
        print(f"intial_annotated_data: total unique peptides in raw annotation data file: {len(annot_data.groupby(['peptide_sequence']))}")

    # should only contain uppercase letters (excludes PTMs)
    meta_data = meta_data[meta_data['modified_sequence'].apply(is_valid_peptide)]

    if VERBOSE:
        print(f"\nptm_filtered_meta_data: number of PSM rows: {len(meta_data)}")

    # Andromeda score indicates how well a candidate peptide sequence explains the observed spectrum, higher means a more confident peptide-spectrum match
    # Spectra instances should have andromeda score of 70 or more
    meta_data = meta_data[meta_data['andromeda_score'] >= 70]

    # If there are spectra having more than one peptide (more than one PSM match), the one is highest andromeda score is selected
    best_andromeda_score_idx = meta_data.groupby(['scan_number', 'raw_file'])['andromeda_score'].idxmax()
    meta_data = meta_data.loc[best_andromeda_score_idx].sort_index()

    if VERBOSE:
        print(f"\nandromeda_filtered_meta_data: number of PSM rows: {len(meta_data)}")

    # Compute peak count per spectrum once, keep only scans with >=20 peaks
    peak_counts = annot_data.groupby(['scan_number', 'raw_file']).size().reset_index(name='n_peaks')
    valid_scans = peak_counts[peak_counts['n_peaks'] >= 20][['scan_number', 'raw_file']]
    meta_data = meta_data.merge(valid_scans, on=['scan_number', 'raw_file'])

    if VERBOSE:
        print(f"20_peak_filtered_meta_data: number of PSM rows: {len(meta_data)}")

    spectra = []

    for i, (_,row) in enumerate(meta_data.iterrows()):

        if i >= max_spectra:
            break
        
        scan = row['scan_number']
        raw_file = row['raw_file']

        scan_annot = annot_data[
            (annot_data['scan_number'] == scan) &
            (annot_data['raw_file'] == raw_file)
        ]

        spectra.append({
            'peptide' : row['modified_sequence'],
            'precursor_mz': row['precursor_mz'],
            'precursor_charge': int(row['precursor_charge']),
            'annotation_rows': scan_annot,
            'scan_number': scan,
            'raw_file': raw_file,
        })

    if VERBOSE:
        print(f"\npre-processed, number of PSMs: {len(spectra)}")
        print(f"pre-processed, number of unique peptides: {len(set(s['peptide'] for s in spectra))}")

    # Create cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(spectra, f)
    if VERBOSE:
        print(f"Saved {len(spectra)} spectra to cache at {cache_path}")

    return spectra

def normalize(s):
    return s.replace('I', 'L')

def process_prospect(annotation_data_path, meta_data_path, output_path, cache_path, num_examples, save_dataset, tolerance, precursor_tolerance):
    spectra = load_prospect(annotation_data_path, meta_data_path, cache_path, num_examples)

    total = 0
    count_label_match = 0
    count_precursor_match = 0
    dataset = []

    for spec in spectra:
        total += 1
        peptide = spec['peptide']

        if VERBOSE:
            print(f"\n=====NEW PEPTIDE=====")
            print(f"label Peptide: {peptide}")
            print(f"tolerance: {tolerance}")
            print(f"precursor_tolerance: {precursor_tolerance}")
            print(f"scan number: {spec['scan_number']}")
            print(f"raw file: {spec['raw_file']}")

        is_label_match, is_precusor_match, predicted_sequence, reasoning_trace = derive_peptide(
            spec['annotation_rows'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge'], tolerance, precursor_tolerance
        )

        if is_label_match: 
            count_label_match += 1
            if VERBOSE:
                print(f"predicted sequence is correctly matches with the label peptide")

        if is_precusor_match: 
            count_precursor_match += 1
            if VERBOSE:
                print(f"predicted sequence is correctly matches with the precursor information")
            
        if save_dataset:
            dataset.append({
            'input': spectrum_to_input_text(
                spec['annotation_rows']['experimental_mass'].tolist(), 
                spec['annotation_rows']['intensity'].tolist(), 
                spec['precursor_mz'], 
                spec['precursor_charge']),
            'output': reasoning_trace,
            'peptide': peptide,
            'scan_number': spec['scan_number'],
            'raw_file': spec['raw_file'],
            'pred_seq': predicted_sequence,
            'is_label_match': is_label_match,
            'is_precusor_match': is_precusor_match
            })
        
            # Store correct peptides
            if total % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Progress: {total} processed")

    print("\nResults:")
    print(f"Total: {total}")
    print(f"correct label matches: {count_label_match} ({100*count_label_match/total:.1f}%)")
    print(f"correct precursor matches: {count_precursor_match} ({100*count_precursor_match/total:.1f}%)")

    if save_dataset:
        print(f"Dataset size: {len(dataset)}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} training examples to {output_path}")

if __name__ == "__main__":

    annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool"
    meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
    OUTPUT_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect_reasoning_dataset_v1.json"
    CACHE_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/100_prospect_cache.pkl"
    TOLERANCE = 0.2
    PRECURSOR_TOLERANCE = 0.1
    TERMINATION_TOLERANCE= 0.01
    VERBOSE = True
    SAVE_DATASET = False
    NUM_EXAMPLES = 100

    process_prospect(annotation_data_path, meta_data_path, OUTPUT_PATH, CACHE_PATH, NUM_EXAMPLES,SAVE_DATASET, TOLERANCE, PRECURSOR_TOLERANCE)