import json
import re
import numpy as np
import pandas as pd

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

# 
131.04049 + 115.02694 + 113.08406 + 129.04259 + 87.03203 + 128.05858 + 1.00728


MDLESQISDMR

NEUTRAL_LOSS_MASSES = {
    'H2O': 18.01056,
    'NH3': 17.02655
}

# MASS_H2O    = 18.01056
# MASS_NH3 = 17.02655
MASS_PROTON =  1.00728
MASS_C13    = 1.00335 # Mass Difference between C-12 and C-13

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
    derived = []
    prev_mz = None

    for i in range(1, n + 1):
        peak_information = get_peak(annotation_lookup, ion_type, i, precursor_charge)
        if VERBOSE:
                print(f"{ion_type}{i}, peak_info: {peak_information}")
        
        if peak_information is not None:
            mz, charge, nl = peak_information
            trace_charge_nl, mass = handle_charge_nl(ion_type, mz, charge, nl, i)

            if is_mass_verified(i, ion_type, mass, ladder_residues, tolerance):
                traces.append(trace_charge_nl)
            else:
                prev_mz = None
                continue
        else:
            prev_mz = None
            continue
        
        if i == 1: 
            first_position_trace = handle_first_position(mass, ion_type, ladder_residues, tolerance)
            traces.append(first_position_trace)

        else:
            if prev_mz is not None:
                predicted_aa_mass = mass-prev_mz
                traces.append(
                        f"Difference between {mass:.5f} and {prev_mz:.5f} gives {predicted_aa_mass:.5f}, "
                        f"Identifying the amino acid residue as {ladder_residues[i-1]}."
                    )
                
            else:
                predicted_aa_mass = mass
                traces.append(
                    f"Identifying the amino acid residue as {ladder_residues[i-1]}."
                )

            prev_mz = mass

        derived.append(ladder_residues[i-1])

    final_seq = ''.join(derived)
    traces.append(f"The complete sequence from the {ion_type}-ion ladder is predicted as {final_seq}.")

    return_final_seq = ''.join(reversed(derived)) if ion_type == 'y' else ''.join(derived)

    return return_final_seq, traces


def derive_peptide(annotation_rows, label_peptide, precursor_mz, precursor_charge, tolerance):

    if VERBOSE:
        print(f"\n=====NEW PEPTIDE=====")
        print(f"Label Peptide: {label_peptide}")
        print(f"Tolerance: {tolerance}")

    reasoning_trace = []
    annotation_lookup = build_annotation_lookup(annotation_rows)

    y_sequences, y_traces = extend_ladder('y', annotation_lookup, label_peptide, precursor_mz, precursor_charge, tolerance)
    y_match = y_sequences == label_peptide
    reasoning_trace.append('\n'.join(y_traces))

    if VERBOSE:
        print(f"Predicted y_seq: {y_sequences}")
        print(f"y_match: {y_match}\n")

    reasoning_trace.append("Now, trying to derive the sequence using the b-ion ladder. ")

    b_sequences, b_traces = extend_ladder('b', annotation_lookup, label_peptide, precursor_mz, precursor_charge, tolerance) 
    b_match = b_sequences == label_peptide
    reasoning_trace.append('\n'.join(b_traces))

    if VERBOSE:
            print(f"Predicted b_seq: {b_sequences}")
            print(f"b_match: {b_match}")

    y_b_match = y_sequences == b_sequences

    if VERBOSE:
        print(f"y_b_match: {y_b_match}")

    if y_b_match:
        reasoning_trace.append(f"Both y-ion and b-ion agree, giving the final derived sequence: {y_sequences}. ")
    else:
        reasoning_trace.append(f"The y-ion ladder gives {y_sequences} while the b-ion ladder gives {b_sequences}. "
                               f"They do not agree, so the sequence could not be confidently cross verified. ")

    complete_resoning_trace = '\n'.join(reasoning_trace)

    return y_sequences, b_sequences, y_match, b_match, y_b_match, complete_resoning_trace

    
def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def spectrum_to_input_text(mz_list, intensity_list, precursor_mz, precursor_charge):
    #  unique_peaks = sorted(set(zip(mz_list, intensity_list)), key=lambda p: p[0])
    peaks = sorted(zip(mz_list, intensity_list), key=lambda p: p[0])
    lines = [f"Precursor mz: {precursor_mz:.5f} Da, Precursor Charge: {precursor_charge}"]
    lines += ["Experimental MS/MS spectrum (m/z, intensity):"]
    lines += [f"m/z {mz:.5f}, intensity {intensity:.5f}" for mz, intensity in peaks]
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

def load_prospect(annotation_data_path, meta_data_path, max_spectra=1):

    meta_data = pd.read_parquet(meta_data_path)
    annot_data = pd.read_parquet(annotation_data_path)

    meta_data = meta_data[meta_data['modified_sequence'].apply(is_valid_peptide)]

    spectra = []

    for i, (_,row) in enumerate(meta_data.iterrows()):
        if i>= max_spectra:
            break
        
        scan = row['scan_number']
        raw_file = row['raw_file']

        scan_annot = annot_data[
            (annot_data['scan_number'] == scan) &
            (annot_data['raw_file'] == raw_file)
        ]

        if scan_annot.empty:
            continue

        spectra.append({
            'peptide' : row['modified_sequence'],
            'precursor_mz': row['precursor_mz'],
            'precursor_charge': int(row['precursor_charge']),
            'annotation_rows': scan_annot,
            'scan_number': scan,
            'raw_file': raw_file,
        })

    return spectra

def normalize(s):
    return s.replace('I', 'L')

def process_prospect(annotation_data_path, meta_data_path, output_path, num_examples, save_dataset, tolerance):
    spectra = load_prospect(annotation_data_path, meta_data_path, num_examples)

    total = 0
    y_correct = 0
    b_correct = 0
    y_b_correct = 0
    dataset = []

    for spec in spectra:
        total += 1
        peptide = spec['peptide']

        y_seq, b_seq, y_match, b_match, y_b_match, reasoning_trace = derive_peptide(
            spec['annotation_rows'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge'], tolerance
        )

        if y_match:
            y_correct +=1
            if VERBOSE:
                print(f"Correct y_sequence match")
                print(f"scan number: {spec['scan_number']}")
                print(f"raw file: {spec['raw_file']}")
        
        if b_match:
            b_correct +=1
            if VERBOSE:
                print(f"Correct b_sequence match")
                print(f"scan number: {spec['scan_number']}")
                print(f"raw file: {spec['raw_file']}")

        if y_b_match: 
            y_b_correct += 1
            if VERBOSE:
                print(f"y and b sequence match is correct")
            
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
            'y_seq': y_seq,
            'b_seq': b_seq,
            'y_match': y_match,
            'b_match': b_match,
            'y_b_match': y_b_match
            })
        
            # Store correct peptides
            if total % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Progress: {total} processed")

    print("\nResults:")
    print(f"Total: {total}")
    print(f"Y correct: {y_correct} ({100*y_correct/total:.1f}%)")
    print(f"B correct: {b_correct} ({100*b_correct/total:.1f}%)")
    print(f"y and b correct: {y_b_correct} ({100*y_b_correct/total:.1f}%)")

    if save_dataset:
        print(f"Dataset size: {len(dataset)}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} training examples to {output_path}")

if __name__ == "__main__":

    annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
    meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
    OUTPUT_PATH = "output/prospect_reasoning_dataset_v1.json"
    TOLERANCE = 0.1
    TERMINATION_TOLERANCE= 0.01
    VERBOSE = True
    SAVE_DATASET = True
    NUM_EXAMPLES = 10

    process_prospect(annotation_data_path, meta_data_path, OUTPUT_PATH,NUM_EXAMPLES,SAVE_DATASET, TOLERANCE)