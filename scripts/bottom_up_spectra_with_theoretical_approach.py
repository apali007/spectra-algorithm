import json
import re
import numpy as np

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

### TODO: verify this
AA_NAMES = {
    'A': 'Alanine (A)', 'R': 'Arginine (R)', 'N': 'Asparagine (N)',
    'D': 'Aspartic acid (D)', 'C': 'Cysteine (C)', 'E': 'Glutamic acid (E)',
    'Q': 'Glutamine (Q)', 'G': 'Glycine (G)', 'H': 'Histidine (H)',
    'I': 'Isoleucine or Leucine (I/L)', 'L': 'Isoleucine or Leucine (I/L)',
    'K': 'Lysine (K)', 'M': 'Methionine (M)', 'F': 'Phenylalanine (F)',
    'P': 'Proline (P)', 'S': 'Serine (S)', 'T': 'Threonine (T)',
    'W': 'Tryptophan (W)', 'Y': 'Tyrosine (Y)', 'V': 'Valine (V)',
}

MASS_H2O    = 18.01056
MASS_PROTON =  1.00728
MASS_C13    = 1.00335 # Mass Difference between C-12 and C-13

def theoretical_ions(peptide, charge=1):
    b_ions, y_ions = {}, {}
    mass = 0.0
    for i in range(len(peptide) -1):
        mass += AA_MASSES[peptide[i]]
        b_ions[f"b{i+1}"] = (mass + charge * MASS_PROTON)/ charge
    mass = 0.0
    for i in range(len(peptide) -1,0,-1):
        mass += AA_MASSES[peptide[i]]
        y_ions[f"y{len(peptide)-i}"] = (mass + MASS_H2O + charge * MASS_PROTON) / charge
    return b_ions, y_ions

def build_resoning_trace(peptide, mz_list, intensity_list, tolerance):
    b_ions, y_ions = theoretical_ions(peptide)
    matched_b, matched_y = [], []
    
    for mz, intensity in zip(mz_list, intensity_list):
        hit = False
        for name, tmz in b_ions.items():
            if abs(mz - tmz) <= tolerance:
                matched_b.append((name, mz, intensity))
                hit = True
                break
        if hit:
            continue
        for name, tmz in y_ions.items():
            if abs(mz-tmz) <=tolerance:
                matched_y.append((name, mz, intensity))
                hit = True
                break
    
    # Sort by ion number
    def ion_num(x):
        return int(re.search(r'\d+', x[0]).group())
    
    matched_b = sorted(matched_b, key=ion_num)
    matched_y = sorted(matched_y, key=ion_num)

    # Need at least 2 matched ions to compute gaps
    if len(matched_y) < 2 and len(matched_b) <2:
        return None
    
    parts = []

    parts.extend(["Input Experimental MS/MS Spectrum",
             f"Step 1 Spectrum summary. Number of peaks {len(mz_list)}.",
             f"mz range {min(mz_list):.2f} to {max(mz_list):.2f}."])
    
    # Y-ion ladder
    if len(matched_y) >= 2:
        parts.append("\n Identifying the y-ion ladder:")
        prev_mz = None
        prev_name = None
        for name, mz, intensity in matched_y:
            if prev_mz is None:
                parts.append(f" {name} ion observed at {mz:.4f}")
            else:
                gap = mz - prev_mz
                best_aa = min(AA_MASSES, key=lambda aa: abs(AA_MASSES[aa] - gap))
                best_diff = abs(AA_MASSES[best_aa] - gap)
                if best_diff <= tolerance:
                    parts.append(
                        f" {name} ion observed at {mz:.4f}. "
                        f"The mass gap from {prev_name} to {name} is {gap:.4f} Da, "
                        f"which corresponds to {AA_NAMES[best_aa]}"
                    )
                else:
                    parts.append(
                        f" {name} ion observed at {mz:.4f}.")
            prev_mz = mz
            prev_name = name
    
    
    # B-ion ladder
    if len(matched_b) >= 2:
        parts.append("\n Identifying the b-ion ladder:")
        prev_mz = None
        prev_name = None
        for name, mz, intensity in matched_b:
            if prev_mz is None:
                parts.append(f" {name} ion observed at {mz:.4f}")
            else:
                gap = mz - prev_mz
                best_aa = min(AA_MASSES, key=lambda aa: abs(AA_MASSES[aa] - gap))
                best_diff = abs(AA_MASSES[best_aa] - gap)
                if best_diff <= tolerance:
                    parts.append(
                        f" {name} ion observed at {mz:.4f}. "
                        f"The mass gap from {prev_name} to {name} is {gap:.4f} Da, "
                        f"which corresponds to {AA_NAMES[best_aa]}"
                    )
                else:
                    parts.append(
                        f" {name} ion observed at {mz:.4f}.")
            prev_mz = mz
            prev_name = name

    parts.append(
        f"\nBased on the identified ion series, "
        f"the predicted peptide sequence is {peptide}"
    )

    return " ".join(parts)

def spectrum_to_input_text(mz_list, intensity_list):
    pairs = sorted(zip(mz_list, intensity_list))
    lines = ["Experimental MS/MS spectrum (m/z, intesity):"]
    lines += [f"m/z {mz:5f} intensity {intensity}" for mz, intensity in pairs]
    lines.append("Analyze this spectrum and identify the most likely peptide sequence.")
    return "\n".join(lines)

def parse_mgf(mgf_path, max_valid_spectra=100):
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
    dataset, skipped = [],0

    for i, spec in enumerate(spectra):
        if num_examples and i >= num_examples:
            break
        peptide = spec.get("peptide")
        if not is_valid_peptide(peptide):
            skipped+=1
            continue
        dataset.append({
            "input": spectrum_to_input_text(spec["mz"], spec["intensity"]),
            "output": build_resoning_trace(peptide, spec["mz"], spec["intensity"], tolerance=tolerance),
            "peptide": peptide,
            "title": spec["title"],
        })

    with open(output_path, "w") as f:
        for ex in dataset:
            f.write(json.dumps({"input": ex["input"], "output": ex["output"]}) + "\n")

    return dataset

if __name__ == "__main__":
    MGF_PATH = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"
    OUTPUT_PATH = "output/bottom_up_spectra_training_data.json"
    TOLERANCE = 0.1
    VERBOSE = False
    NUM_EXAMPLES = None

    process_mgf(MGF_PATH, OUTPUT_PATH, tolerance=TOLERANCE, num_examples=NUM_EXAMPLES)