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

MASS_H2O    = 18.01056
MASS_NH3 = 17.02655
MASS_PROTON =  1.00728
MASS_C13    = 1.00335 # Mass Difference between C-12 and C-13

def extend_ladder(ion_type, mz, intensity, candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance):

    is_y = ion_type == 'y'
    
    precursor_neutral_mass = (precursor_mz * precursor_charge) - (precursor_charge* MASS_PROTON)

    completed_sequences = []

    for start_aa, start_residue_mass in candidates.items():

        sequence = [start_aa] # b1 or y1 initialized
        current_residue_sum = start_residue_mass # b1 or y1 neutral mass initialized
        print(f"Ion Type: {ion_type} | start_candidate_AA: {start_aa}")

        while True:
            if abs(precursor_neutral_mass - current_residue_sum - MASS_H2O) < termination_tolerance:
                print(f" [{ion_type}] Completed: {''.join(sequence)} | residue sum = {current_residue_sum:.4f}")
                completed_sequences.append(''.join(sequence))
                break
            
            next_aa_hits = []
            for next_aa, next_mass in AA_MASSES.items():
                
                for experimental_mz in mz:
                    for charge in range(1, precursor_charge + 1):
                        experimental_mz_neutral = experimental_mz * charge - (charge * MASS_PROTON) 
                        # for nl_mass in [0, MASS_H2O, MASS_NH3]:
                        for nl_mass in [0]:
                            recovered_experimental_mz_mass = experimental_mz_neutral + nl_mass
                            #make singly charged
                            recovered_experimental_mz_mass_singly = recovered_experimental_mz_mass + MASS_PROTON

                            if is_y:
                                expected_mz = current_residue_sum + next_mass + MASS_H2O + MASS_PROTON
                            else:
                                expected_mz = current_residue_sum + next_mass + MASS_PROTON
                            
                            if abs(expected_mz - recovered_experimental_mz_mass_singly) <= tolerance:
                                # print(f"matched experimental_mz = {experimental_mz:.6f} charge={charge} nl={nl_mass:.5f}")
                                next_aa_hits.append((next_aa, next_mass))
            
            next_aa_hits = [('L' if aa == 'I' else aa, mass) for aa, mass in next_aa_hits]
            unique_aa_hits = list({aa: mass for aa, mass in next_aa_hits}.items())

            if len(unique_aa_hits) == 0:
                # check if remaining mass = one AA
                if is_y:
                    remaining_mass = precursor_neutral_mass - current_residue_sum - MASS_H2O
                else:
                    remaining_mass = precursor_neutral_mass - current_residue_sum - MASS_H2O
                
                last_aa_found = False
                for aa, mass in AA_MASSES.items():
                    if abs(remaining_mass-mass) < termination_tolerance:
                        sequence.append(aa)
                        current_residue_sum +=mass
                        print(f"[{ion_type}] Completed: {''.join(sequence)}")
                        completed_sequences.append(''.join(sequence))
                        last_aa_found = True
                       
                if not last_aa_found:
                     print(f"[{ion_type}] Dead end at: {''.join(sequence)}")
                break

            elif len(unique_aa_hits) > 1:

                print(f"[{ion_type}] Ambigous end at: {''.join(sequence)} -> hits: {unique_aa_hits}")
                # print(f" current_residue_sum: {current_residue_sum:.6f}")
                # for hit_aa, hit_mass in unique_aa_hits:
                #     if is_y:
                #         exp = current_residue_sum + hit_mass + MASS_H2O + MASS_PROTON
                #     else:
                #         exp = current_residue_sum + hit_mass + MASS_PROTON
                    # print(f" {hit_aa}: expected_mz={exp:.6f}")
                break
            else:
                next_aa, next_mass = unique_aa_hits[0]
                sequence.append(next_aa)
                current_residue_sum += next_mass

    return completed_sequences


def derive_full_sequence(mz, intensity, y1_candidates, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance):
    y_sequences = extend_ladder('y', mz, intensity, y1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance)
    b_sequences = extend_ladder('b', mz, intensity, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance)
    

    return y_sequences, b_sequences

def derive_peptide(mz, intensity, label_peptide, precursor_mz, precursor_charge, tolerance, termination_tolerance):

    # first derive y1 and b1

    y1_candidates_approach1, b1_candidates_approach1 = find_y1_b1_approach1(mz, intensity, tolerance)

    y1_candidates_approach2, b1_candidates_approach2 = find_y1_b1_approach2(mz, intensity, precursor_mz, precursor_charge,tolerance)

    print(f"\nPeptide: {label_peptide}")
    # print(f" Approach  1 - y1: {y1_candidates_approach1} | b1: {b1_candidates_approach1}")
    # print(f" Approach  2 - y1: {y1_candidates_approach2} | b1: {b1_candidates_approach2}")

    true_y1_aa = label_peptide[-1]
    true_b1_aa = label_peptide[0]
    # print(f" Expected y1: {true_y1_aa} |  Expected b1: {true_b1_aa}")

    y1_candidates = {**y1_candidates_approach1, **y1_candidates_approach2}
    b1_candidates = {**b1_candidates_approach1, **b1_candidates_approach2}

    print(f"y1_candidates: {y1_candidates}")
    print(f"b1_candidates: {b1_candidates}")

    y_sequences, b_sequences = derive_full_sequence(
        mz, intensity, y1_candidates, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance
        )
    
    print(f"y_sequences: {y_sequences}")
    print(f"b_sequences: {b_sequences}")
    print("==============")

    return y_sequences, b_sequences
    
def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def find_y1_b1_approach1(mz, intensity, tolerance):
    y1_candidates = {}
    b1_candidates = {}

    if intensity is None:
        filter_mz = mz
    else:
        filter_mz = filter_top_peaks(mz, intensity,150)

    for aa, mass in AA_MASSES.items():
        y1_expected = mass + MASS_H2O + MASS_PROTON
        b1_expected = mass + MASS_PROTON

        for experimental_mz in filter_mz:

            # Neutral loss
            for nl_mass in [0, MASS_NH3, MASS_H2O]:
                recovered_experimental_mz_mass = experimental_mz + nl_mass

                # singly charge in appraoch 1 because y1 and b1 have small mass containing only one AA. (more charge makes the mass more smaller)
                if abs(recovered_experimental_mz_mass - y1_expected) <=tolerance:
                    # removing the singly charge proton
                    y1_candidates[aa] = experimental_mz - MASS_PROTON - MASS_H2O

                if abs(recovered_experimental_mz_mass - b1_expected) <=tolerance:
                    # removing the singly charge proton
                    b1_candidates[aa] = experimental_mz - MASS_PROTON

    return y1_candidates, b1_candidates

def find_y1_b1_approach2(mz, intensity, precursor_mz, precursor_charge, tolerance):
    y1_candidates = {}
    b1_candidates = {}

    precursor_neutral_mass = (precursor_mz * precursor_charge) - (precursor_charge* MASS_PROTON)
    # make single charged
    precursor_neutral_mass_singly = precursor_neutral_mass + MASS_PROTON
    print(f"\nprecursor_neutral_mass_singly: {precursor_neutral_mass_singly}")
    print(f"precursor_charge: {precursor_charge}")
    
    if intensity is None:
        filter_mz = mz
    else:
        filter_mz = filter_top_peaks(mz, intensity,150)

    # Filtering and keeping only the peaks that is greater than the midpoint of the precursor mass. We only want y(n-1) and b(n-1) here
    mz_midpoint = precursor_mz * 0.5
    print(f"mz_midpoint: {mz_midpoint}")
    large_mz = [m for m in filter_mz if m >= mz_midpoint]

    # mz_array = np.array(mz)
    # threshold = mz_array.max() * 0.2
    # high_mass_peaks = mz_array[mz_array >=threshold]

    for aa, mass in AA_MASSES.items():
        # y1 and b1 ions are singly charged
        y1_expected = mass + MASS_H2O
        b1_expected = mass
        for experimental_mz in large_mz:
            for charge in range(1, precursor_charge+1):

                experimental_mz_neutral = experimental_mz * charge - (charge * MASS_PROTON)  
                for nl_mass in [0, MASS_NH3, MASS_H2O]:
                    recovered_experimental_mz_mass = experimental_mz_neutral + nl_mass
                    #make singly charged
                    recovered_experimental_mz_mass_singly = recovered_experimental_mz_mass + MASS_PROTON
                 
                    derived_AA = precursor_neutral_mass_singly - recovered_experimental_mz_mass_singly

                    if abs(derived_AA - y1_expected) <=tolerance:
                        y1_candidates[aa] = derived_AA - MASS_H2O

                    if abs(derived_AA - b1_expected) <=tolerance:
                        b1_candidates[aa] = derived_AA

    return y1_candidates, b1_candidates

def spectrum_to_input_text(mz_list, intensity_list):
    pairs = sorted(zip(mz_list, intensity_list))
    lines = ["Experimental MS/MS spectrum (m/z, intesity):"]
    lines += [f"m/z {mz:5f} intensity {intensity}" for mz, intensity in pairs]
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

def process_mgf(mgf_path,output_path,tolerance,termination_tolerance, num_examples):
    spectra = parse_mgf(mgf_path)
    # dataset = []

    for i, spec in enumerate(spectra):
        if num_examples and i >= num_examples:
            break
        peptide = spec.get("peptide")
        mz = spec.get("mz")
        intensity = spec.get("intensity")
        pep_mass = spec.get("pepmass")
        pep_charge = spec.get("charge")

        derive_peptide( mz, intensity, peptide, pep_mass, pep_charge, tolerance, termination_tolerance)


def load_prospect(annotation_data_path, meta_data_path,correct_instances_path, max_spectra=1):

    meta_data = pd.read_parquet(meta_data_path)
    annot_data = pd.read_parquet(annotation_data_path)

    meta_data = meta_data[meta_data['modified_sequence'].apply(is_valid_peptide)]

    if correct_instances_path:
        correct_df = pd.read_csv(correct_instances_path)
        meta_data = meta_data.merge(correct_df[['scan_number', 'raw_file']], on=['scan_number', 'raw_file'], how = 'inner')

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

        # mz_list = scan_annot['experimental_mass'].tolist()
        mz_list = scan_annot['theoretical_mass'].tolist()

        spectra.append({
            'peptide' : row['modified_sequence'],
            'precursor_mz': row['precursor_mz'],
            'precursor_charge': int(row['precursor_charge']),
            'mz': mz_list,
            'intensity': None,
            'scan_number': scan,
            'raw_file': raw_file,
        })

    return spectra

def process_prospect(annotation_data_path, meta_data_path, correct_instances_path, tolerance, termination_tolerance, num_examples):
    spectra = load_prospect(annotation_data_path, meta_data_path, correct_instances_path, num_examples)

    total = 0
    y_correct = 0
    b_correct = 0
    both_correct = 0
    # correct_instances = []

    def normalize(s):
        return s.replace('I', 'L')

    for spec in spectra:
        total += 1
        peptide = spec['peptide']
        true_y = normalize(peptide[::-1])
        true_b = normalize(peptide)


        y_seqs, b_seqs = derive_peptide(
            spec['mz'],
            spec['intensity'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge'],
            tolerance,
            termination_tolerance
        )

        y_match = any(normalize(s) == true_y for s in y_seqs)
        b_match = any(normalize(s) == true_b for s in b_seqs)

        if y_match: y_correct +=1
        if b_match: b_correct +=1
        if y_match and b_match: both_correct += 1
        # if y_match or b_match:
        #     correct_instances.append({
        #         'scan_number': spec['scan_number'],
        #         'raw_file':spec['raw_file'],
        #         'peptide':peptide
        #     })

    print("Results:")
    print(f"Total: {total}")
    print(f"Y correct: {y_correct} ({100*y_correct/total:.1f}%)")
    print(f"B correct: {b_correct} ({100*b_correct/total:.1f}%)")
    print(f"Both correct: {both_correct} ({100*both_correct/total:.1f}%)")

    # # Store correct peptides
    # pd.DataFrame(correct_instances).to_csv('output/correct_instances.csv', index=False)
    # print(f"Saved {len(correct_instances)} correct instances")
              


if __name__ == "__main__":

    isProspect = True
    filterCorrectPeptides = True


    if isProspect:
        if filterCorrectPeptides:
            correct_instances_path = "/lclhome/apali007/projects/spectra_algorithm/output/correct_instances.csv"
        else:
            correct_instances_path = ""
        annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
        meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
        OUTPUT_PATH = "output/prospect_bottom_up_spectra_training_data.json"
        TOLERANCE = 0.005
        TERMINATION_TOLERANCE= 0.005
        VERBOSE = False
        NUM_EXAMPLES = 20

        process_prospect(annotation_data_path, meta_data_path, correct_instances_path, tolerance=TOLERANCE, termination_tolerance=TERMINATION_TOLERANCE, num_examples=NUM_EXAMPLES)

    else:
        MGF_PATH = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"
        OUTPUT_PATH = "output/bottom_up_spectra_training_data.json"
        TOLERANCE = 0.02
        TERMINATION_TOLERANCE= 0.01
        VERBOSE = False
        NUM_EXAMPLES = None

        process_mgf(MGF_PATH, OUTPUT_PATH, tolerance=TOLERANCE, termination_tolerance=TERMINATION_TOLERANCE, num_examples=NUM_EXAMPLES)