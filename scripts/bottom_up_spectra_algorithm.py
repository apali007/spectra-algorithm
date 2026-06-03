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

def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def derive_peptide(mz, intensity, label_peptide, precursor_mz, precursor_charge, tolerance):

    y1_candidates_approach1, b1_candidates_approach1 = find_y1_b1_approach1(mz, intensity, tolerance)

    y1_candidates_approach2, b1_candidates_approach2 = find_y1_b1_approach2(mz, intensity, precursor_mz, precursor_charge,tolerance)

    print(f"\nPeptide: {label_peptide}")
    print(f" Approach  1 - y1: {y1_candidates_approach1} | b1: {b1_candidates_approach1}")
    print(f" Approach  2 - y1: {y1_candidates_approach2} | b1: {b1_candidates_approach2}")

    true_y1_aa = label_peptide[-1]
    true_b1_aa = label_peptide[0]
    print(f" Expected y1: {true_y1_aa} |  Expected b1: {true_b1_aa}")

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

def process_mgf(mgf_path,output_path,tolerance,num_examples):
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
        # if not is_valid_peptide(peptide):
        #     continue
        derive_peptide( mz, intensity, peptide, pep_mass, pep_charge, tolerance)


    #     dataset.append({
    #         "input": spectrum_to_input_text(spec["mz"], spec["intensity"]),
    #         "output": build_resoning_trace(peptide, spec["mz"], spec["intensity"], tolerance=tolerance),
    #         "peptide": peptide,
    #         "title": spec["title"],
    #     })

    # with open(output_path, "w") as f:
    #     for ex in dataset:
    #         f.write(json.dumps({"input": ex["input"], "output": ex["output"]}) + "\n")

    # return dataset


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

        # mz_list = scan_annot['experimental_mass'].tolist()
        mz_list = scan_annot['theoretical_mass'].tolist()

        spectra.append({
            'peptide' : row['modified_sequence'],
            'precursor_mz': row['precursor_mz'],
            'precursor_charge': int(row['precursor_charge']),
            'mz': mz_list,
            'intensity': None,
        })

    return spectra

def process_prospect(annotation_data_path, meta_data_path, tolerance, num_examples):
    spectra = load_prospect(annotation_data_path, meta_data_path, num_examples)

    for spec in spectra:
        derive_peptide(
            spec['mz'],
            spec['intensity'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge'],
            tolerance
        )


if __name__ == "__main__":

    isProspect = True

    if isProspect:
        annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
        meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
        OUTPUT_PATH = "output/prospect_bottom_up_spectra_training_data.json"
        TOLERANCE = 0.005
        VERBOSE = False
        NUM_EXAMPLES = 20

        process_prospect(annotation_data_path, meta_data_path, tolerance=TOLERANCE, num_examples=NUM_EXAMPLES)

    else:
        MGF_PATH = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"
        OUTPUT_PATH = "output/bottom_up_spectra_training_data.json"
        TOLERANCE = 0.02
        VERBOSE = False
        NUM_EXAMPLES = None

        process_mgf(MGF_PATH, OUTPUT_PATH, tolerance=TOLERANCE, num_examples=NUM_EXAMPLES)