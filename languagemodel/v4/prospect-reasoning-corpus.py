import json
import re
import numpy as np
import pandas as pd

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

AA_MASSES_WITH_I = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

MASS_H2O    = 18.01056
MASS_NH3 = 17.02655
MASS_PROTON =  1.00728
MASS_C13    = 1.00335 # Mass Difference between C-12 and C-13

def extend_ladder(ion_type, annotation_lookup, label_peptide, precursor_mz, precursor_charge):
    traces = []
    residues = list(label_peptide)
    n = len(residues)
    ladder_residues = list(reversed(residues)) if ion_type == 'y' else residues

    first_aa = ladder_residues[0]
    peak_info = get_peak(annotation_lookup, ion_type, 1)
    obs_mz, z, nl = peak_info

    traces.append(f"Identifying {ion_type}1 from the given spectrum to build the peptide sequence: ")
    if ion_type == 'y':
        residue_mass = obs_mz - MASS_H2O - MASS_PROTON
        traces.append(
            f"mz peak at {obs_mz:.5f} Da. "
            f"Subtracting water ({MASS_H2O:.5f} Da) and a proton ({MASS_PROTON:.5f}) "
            f"gives a residue mass of {residue_mass:.5f} Da, consistent wiht amino acid {first_aa}. "
        )
    else:
        residue_mass = obs_mz - MASS_PROTON
        traces.append(
            f"mz_peak at {obs_mz:.5f} Da. "
            f"Subtracting a proton ({MASS_PROTON:.5f}) "
            f"gives a residue mass of {residue_mass:.5f} Da, consistent with amino acid {first_aa}. "
        )
    
    prev_mz_z1 = obs_mz
    derived = [first_aa]

    for i in range(2, n + 1):
        next_aa = ladder_residues[i-1]
        peak_info = get_peak(annotation_lookup, ion_type, i)
        obs_mz, z, nl = peak_info

        if z == 1:
            obs_mz_z1 = obs_mz
            diff = obs_mz_z1 - prev_mz_z1
            traces.append(
                f"The difference between {ion_type}{i} ({obs_mz_z1:.5f} Da) and "
                f"{ion_type}{i-1} ({prev_mz_z1:.5f} Da) is {diff:.5f} Da, "
                f"Identifying the next residue as {next_aa}. "
            )
        else:
            obs_mz_z1 = obs_mz * z - (z-1) * MASS_PROTON
            diff = obs_mz_z1 - prev_mz_z1
            traces.append(
                f"The {ion_type}{i} is observed at ({obs_mz:.5f} Da) with charge {z}. "
                f"Converting to the singly charged mass of {ion_type}{i} gives {obs_mz_z1:.5f} Da. "
                f"The difference between {ion_type}{i} ({obs_mz_z1:.5f} Da) and "
                f"{ion_type}{i-1} ({prev_mz_z1:.5f} Da) is {diff:.5f} Da, "
                f"identifying the next residue as {next_aa}. "
            )
        
        prev_mz_z1 = obs_mz_z1
        derived.append(next_aa)

    final_seq = ''.join(reversed(derived)) if ion_type == 'y' else ''.join(derived)
    final_seq = normalize(final_seq)
    traces.append(f"The complete sequence from the {ion_type}-ion ladder is predicted as {final_seq}.")

    return final_seq, traces


def derive_peptide(annotation_rows, label_peptide, precursor_mz, precursor_charge):

    if VERBOSE:
        print(f"\nLabel Peptide: {label_peptide}")

    reasoning_trace = []
    annotation_lookup = build_annotation_lookup(annotation_rows)
    normalized_label_peptide = normalize(label_peptide) 

    y_sequences, y_traces = extend_ladder('y', annotation_lookup, label_peptide, precursor_mz, precursor_charge)
    y_match = y_sequences == normalized_label_peptide
    reasoning_trace.append('\n'.join(y_traces))

    if y_match:
        reasoning_trace.append("Successfully built y-ion ladder from this spectrum. Now, trying to derive the sequence using the b-ion ladder. ")
    else:
        reasoning_trace.append("Unable to build y-ion ladder from this spectrum. Now, trying to derive the sequence using the b-ion ladder. ")

    b_sequences, b_traces = extend_ladder('b', annotation_lookup, label_peptide, precursor_mz, precursor_charge) 
    b_match = b_sequences == normalized_label_peptide
    reasoning_trace.append('\n'.join(b_traces))
    
    if b_match:
        reasoning_trace.append("Successfully built b-ion ladder from this spectrum. ")
    else:
        reasoning_trace.append("Unable to build b-ion ladder from this spectrum. ")

    if y_match and b_match:
        reasoning_trace.append(f"Both derived y-ion and b-ion sequence are the same, therefore it is cross-verified and the final sequence derived from the spectrum is {b_sequences}. ")
    else:
        reasoning_trace.append(f"Unable to derive the peptide from this spectrum. ")

    complete_resoning_trace = '\n'.join(reasoning_trace)

    return y_match, b_match, complete_resoning_trace

    
def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def spectrum_to_input_text(mz_list, precursor_mz, precursor_charge):
    mz_sorted = sorted(mz_list)
    lines = [f"Precursor mz: {precursor_mz:.5f} Da, Precursor Charge: {precursor_charge}"]
    lines += ["Experimental MS/MS spectrum (m/z):"]
    lines += [f"m/z {mz:5f}" for mz in mz_sorted]
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
    return bool(peptide) and bool(re.fullmatch(r'[A-Z]+', peptide)) and all(aa in AA_MASSES_WITH_I for aa in peptide)

def get_peak(annotation_lookup, ion_type, position):
    
    k = (ion_type, position, 1, None)
    if k in annotation_lookup:
        return annotation_lookup[k], 1, None
    
    k = (ion_type, position, 2, None)
    if k in annotation_lookup:
        return annotation_lookup[k], 2, None
    
    for nl in ('H2O', 'NH3'):
        k = (ion_type, position, 1, nl)
        if k in annotation_lookup:
            return annotation_lookup[k], 1, nl
        
    for nl in ('H2O', 'NH3'):
        k = (ion_type, position, 2, nl)
        if k in annotation_lookup:
            return annotation_lookup[k], 2, nl
    
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

def process_prospect(annotation_data_path, meta_data_path, output_path, num_examples, save_dataset):
    spectra = load_prospect(annotation_data_path, meta_data_path, num_examples)

    total = 0
    y_correct = 0
    b_correct = 0
    y_and_b_match = 0
    y_or_b_match = 0
    dataset = []

    for spec in spectra:
        total += 1
        peptide = spec['peptide']

        y_match, b_match, reasoning_trace = derive_peptide(
            spec['annotation_rows'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge'],
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

        if y_match or b_match: 
            y_or_b_match += 1
            if VERBOSE:
                print(f"y or b sequence match is correct")
            
        if y_match and b_match: 
            y_and_b_match += 1
            if VERBOSE:
                print(f"y and b sequence match is correct")
            
        if save_dataset:
            dataset.append({
            'input': spectrum_to_input_text(spec['annotation_rows']['experimental_mass'].tolist(), spec['precursor_mz'], spec['precursor_charge']),
            'output': reasoning_trace,
            'peptide': peptide,
            'scan_number': spec['scan_number'],
            'raw_file': spec['raw_file'],
            'y_and_b_match': y_match and b_match
            })
        
            # Store correct peptides
            if total % 1000 == 0:
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Progress: {total} processed, {y_or_b_match} y or b correct so for")

    print("\nResults:")
    print(f"Total: {total}")
    print(f"Y correct: {y_correct} ({100*y_correct/total:.1f}%)")
    print(f"B correct: {b_correct} ({100*b_correct/total:.1f}%)")
    print(f"y and b correct: {y_and_b_match} ({100*y_and_b_match/total:.1f}%)")
    print(f"y or b sequence is correct: {y_or_b_match} ({100*y_or_b_match/total:.1f}%)")

    if save_dataset:
        print(f"Dataset size: {len(dataset)}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} training examples to {output_path}")

if __name__ == "__main__":

    annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
    meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
    OUTPUT_PATH = "output/prospect_reasoning_dataset_v4.json"
    TOLERANCE = 0.005
    TERMINATION_TOLERANCE= 0.01
    VERBOSE = True
    SAVE_DATASET = False
    NUM_EXAMPLES = 1

    process_prospect(annotation_data_path, meta_data_path, OUTPUT_PATH, num_examples=NUM_EXAMPLES, save_dataset = SAVE_DATASET)