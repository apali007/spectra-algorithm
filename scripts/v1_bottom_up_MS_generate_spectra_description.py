import json
import random
import re
import numpy as np

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

MASS_H2O    = 18.01056
MASS_PROTON =  1.00728
MASS_C13    =  1.00335

# Parse MGF file and extract spectrum data
def parse_mgf(mgf_path, max_spectra=100):
    spectra, current, count = [], None, 0
    with open(mgf_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "BEGIN IONS":
                count += 1
                if count > max_spectra:
                    break
                current = {"title": "", "peptide": None, "charge": 1, "pepmass": 0.0, "mz": [], "intensity": []}
            elif line == "END IONS":
                if current is not None:
                    spectra.append(current)
                current = None
            elif current is None:
                continue
            elif line.startswith("TITLE="):
                current["title"] = line[6:]
            elif line.startswith("SEQ="):
                current["peptide"] = ''.join(filter(str.isupper, line[4:]))
            elif line.startswith("PEPMASS="):
                current["pepmass"] = float(line.split("=", 1)[1].split()[0])
            elif line.startswith("CHARGE="):
                try:
                    current["charge"] = int(line.split("=", 1)[1].strip().rstrip("+-"))
                except ValueError:
                    current["charge"] = 1
            elif line[0].isdigit() or line[0] == ".":
                parts = line.split()
                if len(parts) >= 2:
                    current["mz"].append(float(parts[0]))
                    current["intensity"].append(float(parts[1]))
    return spectra

# Process MGF file and save training dataset
def process_mgf(mgf_path, output_path, tolerance=0.05, verbose=True, num_examples=None, top_k=100):
    spectra = parse_mgf(mgf_path)

    for i, spec in enumerate(spectra):
        if num_examples and i >= num_examples:
            break

        mz = spec["mz"]
        intensity = spec["intensity"]
        charge = spec["charge"]
        peptide = spec["peptide"]

        if not mz:
            continue

        top_peaks = get_top_peaks(mz, intensity, top_k)

        # top_masses = [mz_to_neutral_mass(mz_val, charge) for mz_val in top_peaks]

        #charge = 1 and protons cancel out while taking differences so no need to convert to mass
        # top_masses = [mz_to_neutral_mass(mz_val) for mz_val in top_peaks]
        # diffs = compute_peak_differences_from_masses(top_masses)

        diffs = compute_peak_differences_from_masses(top_peaks)

        matches = match_amino_acids(diffs, tolerance)

        if verbose:
            detected_aa = set(extract_aa_sequence(matches))
            true_aa = set(peptide)
            correct_matches = detected_aa & true_aa
            invalid_matches = detected_aa - true_aa
            missed_aa = true_aa - detected_aa


            print(f"\nSpectrum {i+1}")
            print(f"Title: {spec['title']}")
            print(f"Peptide: {spec['peptide']}")
            print(f"Charge: {charge}")
            print(f"Top peaks: {len(top_peaks)}")
            print(f"Peak differences: {len(diffs)}")
            print(f"Number of Matches: {len(matches)}")

            # caution is does not include frequency and sequence context
            print(f"AA Detected Matches: {detected_aa}") # Print unique amino acids detected
            print(f"AA Correct Matches: {correct_matches}") # Print correctly matched amino acids
            print(f"AA Incorrect Matches: {invalid_matches}") # Print incorrectly matched amino acids
            print(f"AA Missed Matches: {missed_aa}") # Print missed amino acids
            print(f"Matches: {matches}") # Print all matches
            

def get_top_peaks(mz_list, intensity_list, top_k=100):
    pairs = list(zip(mz_list, intensity_list))
    pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by intensity

    # no top pair filter
    # top_pairs = pairs[:top_k] # Keep only the top K peaks
    # return [mz for mz, _ in top_pairs] # Return only the m/z values of the top peaks

    return [mz for mz, _ in pairs] # Return only the m/z values of the top peaks

def mz_to_neutral_mass(mz, charge=1):
    return (mz * charge) - (charge * MASS_PROTON)

# def compute_peak_differences_from_masses(mass_list, precision=5):
#     mass_list = sorted(set(round(mass, precision) for mass in mass_list))
#     diffs = set() # Use a set to avoid duplicate differences
#     n = len(mass_list)
#     for i in range(n):
#         for j in range(i + 1, n):
#             diff = abs(mass_list[i] - mass_list[j])
#             if diff > 0:
#                 diffs.add(round(diff, precision))
#     return sorted(diffs)

def compute_peak_differences_from_masses(mass_list, precision=5):
    mass_list = sorted(round(mass, precision) for mass in mass_list)
    diffs = []
    n = len(mass_list)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(mass_list[i] - mass_list[j])
            if diff > 0:
                diffs.append(round(diff, precision))
    return sorted(diffs)

def match_amino_acids(diffs, tolerance=0.1):
    matches = []
    for diff in diffs:
        for aa, mass in AA_MASSES.items():
            if abs(diff - mass) <= tolerance:
                matches.append((aa, diff))
    return matches

def extract_aa_sequence(matches):
    return [aa for aa, _ in matches]

def mgfoutput(MGF_PATH):
    with open(MGF_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i>=1000:
                break
            print(line,end ='')

if __name__ == "__main__":
    MGF_PATH    = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"
    rand_id = random.randint(10000, 99999)
    OUTPUT_PATH = f"output/{rand_id}_spectra_training_data.jsonl"
    TOLERANCE   = 0.1
    VERBOSE     = True
    NUM_EXAMPLES = 50  # set to int to limit
   # NUM_EXAMPLES = None  # set to int to limit

    print(f"\nReading MGF: {MGF_PATH}")
    process_mgf(MGF_PATH, OUTPUT_PATH, tolerance=TOLERANCE, verbose=VERBOSE, num_examples=NUM_EXAMPLES, top_k=100)

    # # Print the top 1000 lines of the mgf file
    # mgfoutput(MGF_PATH)