import json
import re
import numpy as np
import pandas as pd
import glob
import os
import pickle

# Case labels for derive_peptide outcomes
CASE_CLEAN_B_ION_LADDER = "CASE_CLEAN_B_ION_LADDER"
CASE_CLEAN_Y_ION_LADDER = "CASE_CLEAN_Y_ION_LADDER"
CASE_COMPLEMENTARY_GAPS = "CASE_COMPLEMENTARY_GAPS"
CASE_UNRESOLVABLE_MULTI_GAP = "CASE_UNRESOLVABLE_MULTI_GAP"
CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH = "CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH"
CASE_SINGLE_GAP_RESOLVED = "CASE_SINGLE_GAP_RESOLVED"

CASES = [
    CASE_CLEAN_B_ION_LADDER, CASE_CLEAN_Y_ION_LADDER, CASE_COMPLEMENTARY_GAPS, CASE_UNRESOLVABLE_MULTI_GAP, 
    CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH, CASE_SINGLE_GAP_RESOLVED    
]

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

def ordinal(n):
    #Convert an integer to its ordinal string: 1 -> '1st', 2 -> '2nd', 11 -> '11th', etc.
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def format_ladder_sequence_dash(derived_aa):
    seq_str = '-'.join(aa if aa is not None else '?' for aa in derived_aa)
    return seq_str

def format_ladder_sequence(derived_aa):
    seq_str = ''.join(aa if aa is not None else '?' for aa in derived_aa)
    return seq_str

def handle_terminal_ion(ion_type, n, prev_mz, precursor_neutral_mass, predicted_aa_mass, terminal_aa):
    
    terminal_ion_trace = (f"No fragment ion was found for {ion_type}{n}. Since this is the terminal residue, "
        f"it is instead recovered from {ion_type}{n-1} and the precursor mass: precursor-derived neutral mass {precursor_neutral_mass:.5f} Da, "
        f"minus the {ion_type}{n-1} fragment mass ({prev_mz:.5f} Da), plus one proton ({MASS_PROTON:.5f} Da), gives {predicted_aa_mass:.5f} Da. This identifies the residue mass as {terminal_aa}."
        )
    if VERBOSE:
        print(terminal_ion_trace)
    
    return terminal_ion_trace

def validate_precursor(final_sequence, precursor_neutral_mass):

    theoretical_mass_with_water = sum(AA_MASSES[r] for r in final_sequence) +  NEUTRAL_LOSS_MASSES['H2O']
    diff = abs(precursor_neutral_mass - theoretical_mass_with_water)

    if diff > PRECURSOR_TOLERANCE:
        return "", False
    else:
        trace = (f"Validating against the precursor mass: Theoretical mass of {final_sequence} (sum of residue masses plus one water molecule): {theoretical_mass_with_water:.5f} Da. "
        f"Precursor-derived neutral mass: {precursor_neutral_mass:.5f} Da. Difference: {diff:.5f} Da is within the {PRECURSOR_TOLERANCE:.5f} Da tolerance, confirming the derived sequence is consistent with the measured precursor mass. ")

        return trace, True
    

def handle_ion_gaps(b_derived, y_derived, label_peptide, n):
    trace = []
    for i in range(n):
        if b_derived[i] is None:
            if y_derived[i] is not None:
                b_derived[i] = y_derived[i]
                trace.append(f"The residue at position {ordinal(i+1)} (from the N-terminus) was identified by the y-ion ladder as {b_derived[i]}. ")
                if VERBOSE:
                    print(f"The residue at position {ordinal(i+1)} (from the N-terminus) was identified by the y-ion ladder as {b_derived[i]}. ")
            else:
                trace.append(f"The residue at position {ordinal(i+1)} (from the N-terminus) cannot be resolved by the y-ion ladder. ")
                if VERBOSE:
                    print(f"The residue at position {ordinal(i+1)} (from the N-terminus) cannot be resolved by the y-ion ladder. ")

    b_seq_str = format_ladder_sequence_dash(b_derived)
    ion_gap_trace = '\n'.join(trace)

    return b_seq_str, b_derived, ion_gap_trace


def is_mass_verified(i , ion_type, mass , ladder_residues):
    theoretical_mass = sum(AA_MASSES[r] for r in ladder_residues[:i])
    
    if ion_type == 'y':
        theoretical_mass += NEUTRAL_LOSS_MASSES['H2O'] + MASS_PROTON
    else:
        theoretical_mass += MASS_PROTON

    difference = abs(mass - theoretical_mass)
    is_valid = difference < TOLERANCE

    if VERBOSE:
        print(f"{ion_type}{i}, diff: {difference} | is_valid: {is_valid}")

    return is_valid

def handle_charge_nl(ion_type, mz, charge, nl, i):

    # multiple charge with nl
    if charge > 1 and nl is not None:
        singly_charged = mz * charge - (charge - 1) * MASS_PROTON
        correct_mass = singly_charged + NEUTRAL_LOSS_MASSES[nl]
       
        traces = (
                f"Fragment {ion_type}{i} is identified by mz_peak at {mz:.5f} Da with charge {charge} "
                f"and neutral loss of {nl}. Converting into a singly charged ion, it becomes "
                f"{singly_charged:.5f} Da. Correcting for the neutral loss, it becomes {correct_mass:.5f} Da. "
                )
        final_mass = correct_mass

    # multiple charge with no nl
    elif charge > 1:
        singly_charged = mz * charge - (charge - 1) * MASS_PROTON
        traces = (
                f"Fragment {ion_type}{i} is identified by mz_peak at {mz:.5f} Da with charge {charge}. "
                f"Converting into a singly charged ion, it becomes {singly_charged:.5f} Da. ")
        final_mass = singly_charged

    # single charge with nl
    elif nl is not None: 
        correct_mass = mz + NEUTRAL_LOSS_MASSES[nl]
        traces = (
                f"Fragment {ion_type}{i} is identified by mz_peak at {mz:.5f} Da with charge {charge} and neutral loss of {nl}. "
                f"Correcting for the neutral loss, it becomes {correct_mass:.5f} Da. ")
        final_mass = correct_mass

    # single charge with no nl
    else:
        traces = f"Fragment {ion_type}{i} is identified by mz_peak at {mz:.5f} Da. "
        final_mass = mz
    
    return traces, final_mass

def handle_first_position(mz, ion_type, ladder_residues):
    if ion_type == 'y':
        predicted_aa_mass = mz-NEUTRAL_LOSS_MASSES['H2O']-MASS_PROTON
        trace = (
                f"Subtracting H2O ({NEUTRAL_LOSS_MASSES['H2O']:.5f} Da) and a proton ({MASS_PROTON:.5f} Da) "
                f"gives {predicted_aa_mass:.5f} Da. "
                f"This identifies the 1st residue from the C-terminus as {ladder_residues[0]}."
            )
    else:
        predicted_aa_mass = mz-MASS_PROTON
        trace = (
        f"Subtracting a proton ({MASS_PROTON:.5f} Da) gives {mz-MASS_PROTON:.5f} Da. "
        f"This identifies the 1st residue from the N-terminus as {ladder_residues[0]}."
        )
    
    return trace

def extend_ladder(ion_type, annotation_lookup, label_peptide, precursor_neutral_mass, precursor_charge):
    residues = list(label_peptide)
    n = len(residues)
    if ion_type == 'y':
        ladder_residues = list(reversed(residues))  
        terminal = 'C'
    else: 
        ladder_residues = residues
        terminal = 'N'

    traces = []

    # Initializing the position of fragment ions to be derived with none.
    derived_aa_pos = [None] * n
    prev_mz = None

    for i in range(1, n + 1):
        peak_information = get_peak(annotation_lookup, ion_type, i, precursor_charge)
        if VERBOSE:
                print(f"{ion_type}{i}, peak_info: {peak_information}")
        
        if peak_information is None:
            if i == n and prev_mz is not None:
                if ion_type == 'y':
                    predicted_aa_mass = precursor_neutral_mass - prev_mz + MASS_PROTON
                else:
                    predicted_aa_mass = precursor_neutral_mass - NEUTRAL_LOSS_MASSES['H2O'] - prev_mz + MASS_PROTON

                diff = abs(predicted_aa_mass - AA_MASSES[ladder_residues[i-1]])
                if diff < TOLERANCE:
                    traces.append(handle_terminal_ion(ion_type, n, prev_mz, precursor_neutral_mass, predicted_aa_mass, ladder_residues[i-1]))
                    derived_aa_pos[i-1] = ladder_residues[i-1] 
                else:
                    traces.append(f"No fragment ion was found for {ion_type}{i}. ")
            else:
                traces.append(f"No fragment ion was found for {ion_type}{i}. ")
            prev_mz = None
            continue

        mz, charge, nl = peak_information
        trace_charge_nl, mass = handle_charge_nl(ion_type, mz, charge, nl, i)

        if not is_mass_verified(i, ion_type, mass, ladder_residues):
            traces.append(f"No fragment ion was found for {ion_type}{i}. ")
            prev_mz = None
            continue

        traces.append(trace_charge_nl)          
        
        if i == 1: 
            traces.append(handle_first_position(mass, ion_type, ladder_residues))
            derived_aa_pos[i-1] = ladder_residues[i-1]
        else:
            if prev_mz is not None:
                predicted_aa_mass = mass-prev_mz
                traces.append(
                        f"The difference between {ion_type}{i} and {ion_type}{i-1} ( {mass:.5f} - {prev_mz:.5f} = {predicted_aa_mass:.5f} Da) identifies the {ordinal(i)} residue from the {terminal}-terminus as {ladder_residues[i-1]}. "
                    )
                derived_aa_pos[i-1] = ladder_residues[i-1]
            # else:
            #     # assign amino acid if prev_mz is none but the peak exists in the labeled dataset. But you cannot compute the difference ss, you just use the labeled peptide position.
            #     traces.append(
            #     f"This identifies the {ordinal(i)} residue from the {terminal}-terminus as {ladder_residues[i-1]}. "
            #         )
            #     derived_aa_pos[i-1] = ladder_residues[i-1]


        prev_mz = mass

    if VERBOSE:
        detected_positions = [f"{ion_type}{i+1}" for i, aa in enumerate(derived_aa_pos) if aa is not None]
        gap_positions = [f"{ion_type}{i+1}" for i, aa in enumerate(derived_aa_pos) if aa is None]
        print(f"detected_positions: {', '.join(detected_positions)}")
        print(f"gap_positions: {', '.join(gap_positions)}")

    derived_aa_pos = list(reversed(derived_aa_pos)) if ion_type == 'y' else derived_aa_pos  
        
    return derived_aa_pos, traces

def derive_peptide(annotation_rows, label_peptide, precursor_mz, precursor_charge):
    label_residues = list(label_peptide)
    precursor_neutral_mass = precursor_mz * precursor_charge - precursor_charge * MASS_PROTON
    n = len(label_peptide)
    if VERBOSE:
        print(f"Length of the peptide: {n}")
        print(f"precursor_mz: {precursor_mz:.5f}")
        print(f"precursor_mz: {precursor_charge}")
    reasoning_trace = []
    annotation_lookup = build_annotation_lookup(annotation_rows)

    reasoning_trace.append("Analyzing spectra for the y-ion and b-ion ladder to predict the peptide sequence, using the given precursor information. ")

    reasoning_trace.append("First, trying to derive the sequence using the y-ion ladder (C-terminus to N-terminus): ")
    y_derived, y_traces = extend_ladder('y', annotation_lookup, label_peptide, precursor_neutral_mass, precursor_charge)
    reasoning_trace.append('\n'.join(y_traces))

    y_seq_str = format_ladder_sequence_dash(y_derived)
    y_has_gap = any(aa is None for aa in y_derived)

    reasoning_trace.append(f"The y-ion ladder identifies this sequence (N-terminus to C-terminus): {y_seq_str}. ")

    if not y_has_gap:
        final_sequence = "".join(y_derived)
        precursor_validation_trace, is_precursor_match = validate_precursor(final_sequence, precursor_neutral_mass)
        reasoning_trace.append(precursor_validation_trace)
        reasoning_trace.append(f"No need to derive a b-ion ladder as the complete sequence is confirmed. Final predicted sequence: {final_sequence}. ")
        is_match = (label_peptide == final_sequence)
        return is_match, is_precursor_match, final_sequence, reasoning_trace, CASE_CLEAN_Y_ION_LADDER

    reasoning_trace.append("Next, trying to derive the sequence using the b-ion ladder (N-terminus to C-terminus): ")
    b_derived, b_traces = extend_ladder('b', annotation_lookup, label_peptide, precursor_neutral_mass, precursor_charge) 
    reasoning_trace.append('\n'.join(b_traces))

    b_seq_str = format_ladder_sequence_dash(b_derived)
    b_has_gap = any(aa is None for aa in b_derived)

    reasoning_trace.append(f"The b-ion ladder identifies this sequence (N-terminus to C-terminus): {b_seq_str}. ")
    
    if not b_has_gap:
        final_sequence = "".join(b_derived)
        precursor_validation_trace, is_precursor_match = validate_precursor(final_sequence, precursor_neutral_mass)
        reasoning_trace.append(precursor_validation_trace)
        reasoning_trace.append(f"Final predicted sequence: {final_sequence}. ")
        is_match = (label_peptide == final_sequence)
        return is_match, is_precursor_match, final_sequence, reasoning_trace, CASE_CLEAN_B_ION_LADDER

    reasoning_trace.append("Resolving the remaining b-ion ladder gaps using the y-ion ladder:")
    b_seq_str_after_handle, b_derived_after_handle, ion_gap_trace = handle_ion_gaps(b_derived, y_derived, label_peptide, n)
    reasoning_trace.append(ion_gap_trace)

    
    gap_indices = [i for i, aa in enumerate(b_derived_after_handle) if aa is None]
    gap_count = len(gap_indices)    

    if gap_count == 0:
        final_sequence_after_handle = "".join(b_derived_after_handle)
        reasoning_trace.append(f"Combining both ladders gives the full predicted sequence: {final_sequence_after_handle}. ")
        precursor_validation_trace, is_precursor_match = validate_precursor(final_sequence_after_handle, precursor_neutral_mass)
        reasoning_trace.append(precursor_validation_trace)
        reasoning_trace.append(f"Final predicted sequence: {final_sequence_after_handle}. ")
        is_match = (label_peptide == final_sequence_after_handle)
        return is_match, is_precursor_match, final_sequence_after_handle, reasoning_trace, CASE_COMPLEMENTARY_GAPS

    if gap_count == 1:
        gap_pos = gap_indices[0]
        aa_gap = label_residues[gap_pos]
        sum_identified_residues = sum(AA_MASSES[aa] for aa in b_derived_after_handle if aa is not None)
        identified_residues = '+'.join(aa for aa in b_derived_after_handle if aa is not None)
        predicted_aa_mass = precursor_neutral_mass - sum_identified_residues - NEUTRAL_LOSS_MASSES['H2O']
        diff = abs(predicted_aa_mass - AA_MASSES[aa_gap])
        if diff < TOLERANCE:
            incomplete_sequence = format_ladder_sequence(b_derived_after_handle) 
            reasoning_trace.append(
            f"Combining both ladders gives the incomplete predicted sequence: {incomplete_sequence}. "
            f"Since exactly one residue remains unresolved, it can be recovered directly from the precursor mass: "
            f"precursor-derived neutral mass {precursor_neutral_mass:.5f} Da, minus water ({NEUTRAL_LOSS_MASSES['H2O']:.5f} Da), minus the sum of the already-identified residues ({identified_residues} = {sum_identified_residues:.5f} Da), "
            f"leaves {predicted_aa_mass:.5f} Da, matching the mass of {aa_gap}. This identifies the residue at position {gap_pos+1} as {aa_gap}."
            )

            b_derived_after_handle[gap_pos] = aa_gap 
            final_sequence_after_handle = "".join(b_derived_after_handle)

            reasoning_trace.append(f"Combining all resolved positions gives the complete predicted sequence: {final_sequence_after_handle}. ")

            precursor_validation_trace, is_precursor_match = validate_precursor(final_sequence_after_handle, precursor_neutral_mass)
            reasoning_trace.append(precursor_validation_trace)
            reasoning_trace.append(f"Final predicted sequence: {final_sequence_after_handle}. ")
            is_match = (label_peptide == final_sequence_after_handle)
            return is_match, is_precursor_match, final_sequence_after_handle, reasoning_trace, CASE_SINGLE_GAP_RESOLVED
            
        else:
            final_sequence_after_handle = format_ladder_sequence(b_derived_after_handle) 
            reasoning_trace.append(f"Combining both ladders gives the incomplete predicted sequence: {final_sequence_after_handle}. The complete sequence cannot be predicted. ")
            reasoning_trace.append(f"No final sequence is reported for this spectrum. ")
            is_match = (label_peptide == final_sequence_after_handle)
            return is_match, False, final_sequence_after_handle, reasoning_trace, CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH

    else:
        
        final_sequence_after_handle = format_ladder_sequence(b_derived_after_handle) 
        reasoning_trace.append(f"Combining both ladders gives the incomplete predicted sequence: {final_sequence_after_handle}. The complete sequence cannot be predicted. ")
        reasoning_trace.append(f"No final sequence is reported for this spectrum. ")
        is_match = (label_peptide == final_sequence_after_handle)
        return is_match, False, final_sequence_after_handle, reasoning_trace, CASE_UNRESOLVABLE_MULTI_GAP
    
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

def load_prospect(annotation_data_path, meta_data_path, cache_path, max_spectra, use_cache):

    if use_cache:
        # check if cache directory exists (100_prospect_cache.pkl)
        if os.path.exists(cache_path):
            if VERBOSE:
                print(f"Found cache at {cache_path}, loading instead of reprocessing")
            with open(cache_path, 'rb') as f:
                spectra = pickle.load(f)
            return spectra[:max_spectra]


    meta_data = pd.read_parquet(meta_data_path)
    annotation_files = sorted(glob.glob(f"{annotation_data_path}/TUM_third_pool_*_01_01_annotation.parquet"))
    if VERBOSE:
        print(f"Found {len(annotation_files)} annotation files: {annotation_files}")

    annot_data = pd.concat(
        [pd.read_parquet(f) for f in annotation_files],
        ignore_index = True
    )

    if VERBOSE:
        print(f"initial_meta_data: number of PSM rows, len(meta_data): {len(meta_data)}")
        print(f"initial_meta_data: total unique spectra in raw meta data file, meta_data.groupby(['scan_number', 'raw_file']): {len(meta_data.groupby(['scan_number', 'raw_file']))}")
        print(f"initial_meta_data: total unique peptides in raw meta data file, len(meta_data.groupby(['modified_sequence'])): {len(meta_data.groupby(['modified_sequence']))}")
        print(f"initial_meta_data: total unique spectra and peptide pairs in raw meta data file, len(meta_data.groupby(['scan_number', 'raw_file', 'modified_sequence'])): {len(meta_data.groupby(['scan_number', 'raw_file', 'modified_sequence']))}")
        print(f"\nintial_annotated_data: total unique spectra in raw annotation file, len(annot_data.groupby(['scan_number', 'raw_file'])): {len(annot_data.groupby(['scan_number', 'raw_file']))}")
        print(f"intial_annotated_data: total unique peptides in raw annotation data file, len(annot_data.groupby(['peptide_sequence'])): {len(annot_data.groupby(['peptide_sequence']))}")
        print(f"\nintial_annotated_data: total unique spectra and peptide pairs in raw annotation file, len(annot_data.groupby(['scan_number', 'raw_file', 'peptide_sequence'])): {len(annot_data.groupby(['scan_number', 'raw_file', 'peptide_sequence']))}")

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

    filtered_annot = annot_data.merge(
        meta_data[['scan_number', 'raw_file']],
        on=['scan_number', 'raw_file'])

    unique_triples = filtered_annot[['scan_number', 'raw_file', 'peptide_sequence']].drop_duplicates()
    unique_pairs = filtered_annot[['scan_number', 'raw_file']].drop_duplicates()

    if VERBOSE:
        print(f"\n Ambiguity check (on filtered spectra only):")
        print(f" unique (scan, raw_file, peptide) triples: {len(unique_triples)}")
        print(f" unique (scan, raw_file) pairs: {len(unique_pairs)}")
        print(f" difference (ambigous scans) : {len(unique_triples) - len(unique_pairs)}")

    annot_index = filtered_annot.groupby(['scan_number', 'raw_file', 'peptide_sequence']).indices
    
    spectra = []

    for i, row in enumerate(meta_data.itertuples(index=False)):
        
        scan = row.scan_number
        raw_file = row.raw_file
        peptide = row.modified_sequence

        idx = annot_index.get((scan, raw_file, peptide))

        if idx is None:
            continue

        scan_annot = filtered_annot.iloc[idx]

        spectra.append({
            'peptide' : peptide,
            'precursor_mz': row.precursor_mz,
            'precursor_charge': int(row.precursor_charge),
            'annotation_rows': scan_annot,
            'scan_number': scan,
            'raw_file': raw_file,
        })

    if VERBOSE:
        print(f"\npre-processed, number of PSMs: {len(spectra)}")
        print(f"pre-processed, number of unique peptides: {len(set(s['peptide'] for s in spectra))}")

    if use_cache:
        # Create cache directory
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(spectra, f)
        if VERBOSE:
            print(f"Saved {len(spectra)} spectra to cache at {cache_path}")

    return spectra[:max_spectra]

def normalize(s):
    return s.replace('I', 'L')

def process_prospect(annotation_data_path, meta_data_path, output_path, cache_path, num_examples, save_dataset, use_cache):
    spectra = load_prospect(annotation_data_path, meta_data_path, cache_path, num_examples, use_cache)

    total = 0
    count_label_match = 0
    count_precursor_match = 0
    dataset = []
    case_counts = {}

    for spec in spectra:
        total += 1
        peptide = spec['peptide']

        if VERBOSE:
            print(f"\n=====NEW PEPTIDE=====")
            print(f"label Peptide: {peptide}")
            print(f"tolerance: {TOLERANCE}")
            print(f"precursor_tolerance: {PRECURSOR_TOLERANCE}")
            print(f"scan number: {spec['scan_number']}")
            print(f"raw file: {spec['raw_file']}")

        is_label_match, is_precusor_match, predicted_sequence, reasoning_trace, case_label = derive_peptide(
            spec['annotation_rows'],
            spec['peptide'],
            spec['precursor_mz'],
            spec['precursor_charge']
        )

        complete_resoning_trace = '\n'.join(reasoning_trace)

        if VERBOSE:
            print(f"is_label_match: {is_label_match}")
            print(f"is_precusor_match: {is_precusor_match}")
            print(f"predicted_sequence: {predicted_sequence}")
            print(f"case_label: {case_label}")
        

        if is_label_match: 
            count_label_match += 1
            if VERBOSE:
                print(f"predicted sequence is correctly matches with the label peptide")

        if is_precusor_match: 
            count_precursor_match += 1
            if VERBOSE:
                print(f"predicted sequence is correctly matches with the precursor information")

        case_counts[case_label] = case_counts.get(case_label, 0) + 1
            
        if save_dataset:
            dataset.append({
            'input': spectrum_to_input_text(
                spec['annotation_rows']['experimental_mass'].tolist(), 
                spec['annotation_rows']['intensity'].tolist(), 
                spec['precursor_mz'], 
                spec['precursor_charge']),
            'output': complete_resoning_trace,
            'peptide': peptide,
            'scan_number': spec['scan_number'],
            'raw_file': spec['raw_file'],
            'pred_seq': predicted_sequence,
            'is_label_match': is_label_match,
            'is_precusor_match': is_precusor_match,
            'case_label': case_label
            })
        
            # Store correct peptides
            if total % 1000 == 0:
                with open(output_path, 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Progress: {total} processed")

    print("\nResults:")
    print(f"Total: {total}")
    print(f"correct label matches: {count_label_match} ({100*count_label_match/total:.1f}%)")
    print(f"correct precursor matches: {count_precursor_match} ({100*count_precursor_match/total:.1f}%)")

    print("\nCase Counts:")
    for case_label in CASES:
        count = case_counts.get(case_label, 0)
        print(f"{case_label}: {count} ({100*count/total:.1f}%)")

    if save_dataset:
        print(f"Dataset size: {len(dataset)}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} training examples to {output_path}")

if __name__ == "__main__":

    annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool"
    meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
    OUTPUT_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/output/prospect_reasoning_dataset_v1.json"
    CACHE_PATH_100 = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/100_prospect_cache.pkl"
    CACHE_PATH_ALL_SPECTRA = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_spectra_prospect_cache.pkl"
    TOLERANCE = 0.2
    PRECURSOR_TOLERANCE = 0.1
    VERBOSE = False
    SAVE_DATASET = False
    NUM_EXAMPLES = 1
    use_cache = True

    process_prospect(annotation_data_path, meta_data_path, OUTPUT_PATH, CACHE_PATH_ALL_SPECTRA, NUM_EXAMPLES,SAVE_DATASET, use_cache)