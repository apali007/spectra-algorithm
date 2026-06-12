import json
import re
import numpy as np
import pandas as pd
import os

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

def emit_candidates_summary(ion_type, candidates):
    ion_label = "y1" if ion_type == 'y' else "b1"

    if not candidates:
        return (
            f"No {ion_label} candidates found in the given spectrum. "
            f"Unable to derive the peptide sequence from this spectrum."
        )
    
    lines = [f"Identifying {ion_label} candidates from the given spectrum to build the peptide sequence: "]

    for aa, (residue_mass, mz, charge, nl_mass, from_precursor) in candidates.items():
        lines.append(
            
            f"mz peak at {mz:.5f} Da. Subtracting water ({MASS_H2O} Da) and a proton ({MASS_PROTON})"
            f" gives a residue mass of {residue_mass:.5f} Da, "
            f"consistent with amino acid {aa}."
        )

    candidates_aa = list(candidates.keys())
    lines.append(f"The observed {ion_label} candidates: {', '.join(candidates_aa)}")

    return "\n".join(lines)


def extend_ladder(ion_type, mz, intensity, candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance):

    is_y = ion_type == 'y'

    if is_y:
        terminal = 'C'
    else:
        terminal = 'N'
    
    precursor_neutral_mass = (precursor_mz * precursor_charge) - (precursor_charge* MASS_PROTON)
    
    completed_sequences = []
    completed_traces = []
    
    found_peptide_sequence = False
    
    if VERBOSE:
        print(f"Final selected candidates -> Ion type: {ion_type} candidates: ", candidates.items())

    for start_aa, (start_residue_mass, start_mz, start_charge, start_nl, from_precursor) in candidates.items():
        completed_sequences = []
        completed_traces = []
        trace_lines = []
        position = 1
        sequence = [start_aa] # b1 or y1 initialized
        current_residue_sum = start_residue_mass # b1 or y1 neutral mass initialized
        current_mz = start_mz
        trace_lines.append(f"Attempting {ion_type}-ion ladder starting with {start_aa} as the {terminal}-terminal amino acid ({ion_type}{position} = {start_mz:.5f} Da):")
        
        if VERBOSE:
            print(f"Attempting {ion_type}-ion ladder starting with {start_aa} as the {terminal}-terminal amino acid ({ion_type}{position} = {start_mz:.5f} Da):")
        
        
        # trace_lines.append(emit_trace_line(ion_type, position, start_mz, start_aa, start_residue_mass, start_charge, start_nl, from_precursor))

        while True:
            #overshot pre cursor mass
            if (precursor_neutral_mass - current_residue_sum - MASS_H2O) < 0:
                trace_lines.append(
                    f"No peak found for {ion_type}{position+1}. "
                    f"{ion_type} ladder terminated at position {ion_type}{position}. "
                    f"Unable to derive the peptide sequence from this spectrum."
                )
                completed_traces.append('\n'.join(trace_lines))
                break
            
            next_aa_hits = []
            for next_aa, next_mass in AA_MASSES.items():
                
                for experimental_mz in mz:
                    for charge in range(1, precursor_charge + 1):
                    # for charge in range(2):
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
                                next_aa_hits.append((next_aa, next_mass, experimental_mz, recovered_experimental_mz_mass_singly, charge, nl_mass))
            
            next_aa_hits = [('L' if aa == 'I' else aa, mass, exp_mz, rec_mz, ch, nl) for aa, mass, exp_mz, rec_mz, ch, nl in next_aa_hits]
            unique_aa_hits = list({aa: (mass, exp_mz, rec_mz, ch, nl) for aa, mass, exp_mz, rec_mz, ch, nl in next_aa_hits}.items())

            if len(unique_aa_hits) == 0:
                # check if remaining mass = one AA
                remaining_mass = precursor_neutral_mass - current_residue_sum - MASS_H2O
        
                last_aa_found = False
                for aa, mass in AA_MASSES.items():
                    if abs(remaining_mass-mass) < termination_tolerance:
                        sequence.append(aa)
                        peptide = ''.join(sequence)

                        current_residue_sum +=mass
                        # print(f"[{ion_type}] Completed: {''.join(sequence)}")
                        if precursor_charge > 1:
                            precursor_mz_singly = precursor_mz * precursor_charge - (precursor_charge * MASS_PROTON) + MASS_PROTON
                            trace_lines.append(
                                f"No peak found for {ion_type}{position+1}. "
                                f"Using precursor mass ({precursor_mz:.5f} Da, charge={precursor_charge}), "
                                f"the singly charged precursor mass is {precursor_mz_singly:.5f} Da, the mz for ion {ion_type}{position} is {current_mz:.5f} Da. "
                                f"The difference between them is {precursor_mz_singly-current_mz:.5f} Da, which is consistent with amino acid {aa}. "
                                f"The complete sequence is predicted as {peptide[::-1]}."
                            )
                        else:
                            trace_lines.append(
                                f"No peak found for {ion_type}{position+1}. "
                                f"Using precursor mass ({precursor_mz:.5f} Da, charge={precursor_charge}) and "
                                f"the mz for ion {ion_type}{position} ({current_mz:.5f} Da), "
                                f"the difference between them is {precursor_mz-current_mz:.5f} Da, which is consistent with amino acid {aa}. "
                                f"The complete sequence is predicted as {peptide[::-1]}."
                            )

                        completed_sequences.append(peptide[::-1])
                        completed_traces.append('\n'.join(trace_lines))
                        last_aa_found = True
                        found_peptide_sequence = True
                        break
                       
                if not last_aa_found:
                    #  print(f"[{ion_type}] Dead end at: {''.join(sequence)}")
                     trace_lines.append(
                        f"No peak found for {ion_type}{position+1}. "
                        f"{ion_type} ladder terminated at position {ion_type}{position}. "
                        f"Unable to derive the peptide sequence from this spectrum."
                     )
                     completed_traces.append('\n'.join(trace_lines))
                break

            elif len(unique_aa_hits) > 1:

                # print(f"[{ion_type}] Ambigous end at: {''.join(sequence)} -> hits: {unique_aa_hits}")

                # print(f" current_residue_sum: {current_residue_sum:.6f}")
                # for hit_aa, hit_mass in unique_aa_hits:
                #     if is_y:
                #         exp = current_residue_sum + hit_mass + MASS_H2O + MASS_PROTON
                #     else:
                #         exp = current_residue_sum + hit_mass + MASS_PROTON
                    # print(f" {hit_aa}: expected_mz={exp:.6f}")
                trace_lines.append(
                    f"Multiple residues match at position {ion_type}{position+1}. "
                    f"Checking mass differences from {ion_type}{position} ({current_mz:.5f} Da). "
                )
                for hit_aa, (hit_res_mass, hit_exp_mz, hit_rec_mz, hit_charge, hit_nl) in unique_aa_hits:
                    if hit_charge > 1:
                        trace_lines.append(
                            f"Peak at {hit_exp_mz:.5f} Da (charge={hit_charge}), "
                            f"converted to singly charged: {hit_rec_mz:.5f} Da. "
                            f"Difference from {ion_type}{position} ({current_mz:.5f} Da): "
                            f"{hit_rec_mz - current_mz:.5f} Da, consistent with amino acid {hit_aa}. "
                        )
                    else:
                        trace_lines.append(
                            f"Peak at {hit_exp_mz:.5f} Da (charge={hit_charge}), "
                            f"Difference from {ion_type}{position} ({current_mz:.5f} Da): "
                            f"{hit_exp_mz - current_mz:.5f} Da, consistent with amino acid {hit_aa}. "
                        )
                trace_lines.append(
                    f"Ambiguous, cannot distinguish between candidates. "
                    f"{ion_type} ladder terminated at position {ion_type}{position}. "
                    f"Unable to derive the peptide sequence from this spectrum."
                )
                completed_traces.append('\n'.join(trace_lines))
                break
            else:
                next_aa, (next_mass, experimental_mz, recovered_experimental_mz_mass_singly, charge, nl_mass) = unique_aa_hits[0]
                position += 1
                if charge > 1:
                    trace_lines.append(
                    f"The y{position} is observed at ({experimental_mz:.5f} Da) with charge {charge}. Converting to the singly charged mass of {ion_type}{position} gives {recovered_experimental_mz_mass_singly:.5f} Da."
                    f" The difference between {ion_type}{position} ({recovered_experimental_mz_mass_singly:.5f} Da) and"
                    f" y{position-1} ({current_mz:.5f} Da) is {next_mass:.5f} Da,"
                    f" identifying the next residue as {next_aa}."
                    )
                    current_mz = recovered_experimental_mz_mass_singly
                else:
                    trace_lines.append(
                    f"The difference between y{position} ({experimental_mz:.5f} Da) and "
                    f"y{position-1} ({current_mz:.5f} Da) is {next_mass:.5f} Da, "
                    f"identifying the next residue as {next_aa}."
                    )
                    current_mz = experimental_mz
                
                sequence.append(next_aa)
                current_residue_sum += next_mass

                if abs(current_residue_sum + MASS_H2O - precursor_neutral_mass) <= termination_tolerance:
                        peptide = ''.join(sequence)
                        trace_lines.append(
                                f"The complete sequence is predicted as {peptide[::-1]}."
                            )
                        completed_sequences.append(peptide[::-1])
                        completed_traces.append('\n'.join(trace_lines))
                        found_peptide_sequence = True
                        break
                
        if found_peptide_sequence:
            if VERBOSE:
                print(f"Found the peptide sequence with start_aa ({start_aa}), the sequence is {completed_sequences}.")
            break
    
    return completed_sequences, completed_traces


def derive_full_sequence(mz, intensity, y1_candidates, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance):
    y_sequences, y_traces = extend_ladder('y', mz, intensity, y1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance)
    # b_sequences, b_traces = extend_ladder('b', mz, intensity, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance)
    b_sequences = {}
    b_traces = []
    

    return y_sequences, y_traces, b_sequences, b_traces

def derive_peptide(mz, intensity, label_peptide, precursor_mz, precursor_charge, tolerance, termination_tolerance):

     
    is_y1_candidates_direct = False
    is_b1_candidates_direct = False
    is_y1_candidates_back_calc = True
    is_b1_candidates_back_calc = False

    reasoning_trace = []

    # first derive y1 and b1


    if is_y1_candidates_direct or is_b1_candidates_direct:
        y1_candidates_direct, b1_candidates_direct = find_y1_b1_direct(mz, intensity, tolerance)

    if not is_y1_candidates_direct:    
        y1_candidates_direct = {}

    if not is_b1_candidates_direct:
        b1_candidates_direct = {}

    reasoning_trace.append(emit_candidates_summary('y', y1_candidates_direct))
    
    b1_candidates_direct = {}

    if is_y1_candidates_back_calc or is_b1_candidates_back_calc:
        y1_candidates_back_calc, b1_candidates_back_calc = find_y1_b1_back_calc(mz, intensity, precursor_mz, precursor_charge,tolerance)

    if not is_y1_candidates_back_calc:    
        y1_candidates_back_calc = {}

    if not is_b1_candidates_back_calc:
        b1_candidates_back_calc = {}


    if VERBOSE:
        print(f"\nLabel Peptide: {label_peptide}")
        print(f" Direct Approach - y1: {y1_candidates_direct} | b1: {b1_candidates_direct}")
        print(f" Back calc Approach - y1: {y1_candidates_back_calc} | b1: {b1_candidates_back_calc}")

    true_y1_aa = label_peptide[-1]
    true_b1_aa = label_peptide[0]
    # print(f" Expected y1: {true_y1_aa} |  Expected b1: {true_b1_aa}")

    # Prioritize direct approach over back calc approach. If the AA are same, the direct approach AA is picked.
    y1_candidates = dict(y1_candidates_direct)
    for aa, val in y1_candidates_back_calc.items():
        if aa not in y1_candidates:
            y1_candidates[aa] = val

    b1_candidates = dict(b1_candidates_direct)
    for aa, val in b1_candidates_back_calc.items():
        if aa not in b1_candidates:
            b1_candidates[aa] = val

    # if VERBOSE:
    #     print(f"y1_candidates: {y1_candidates}")
    #     print(f"b1_candidates: {b1_candidates}")

    y_sequences, y_traces, b_sequences, b_traces = derive_full_sequence(
        mz, intensity, y1_candidates, b1_candidates, precursor_mz, precursor_charge, tolerance, termination_tolerance
        )
    
    reasoning_trace.append('\n'.join(y_traces))
    complete_resoning_trace = '\n'.join(reasoning_trace)
    # print(f"y_sequences: {y_sequences}")
    # print(f"b_sequences: {b_sequences}")
    
    # if VERBOSE:
        # print(f"Reasoning trace")
        # print(complete_resoning_trace)
        # print("==============")

    return y_sequences, b_sequences, complete_resoning_trace
    
def filter_top_peaks(mz, intensity, top_n):
    mz_array = np.array(mz)
    intensity_array = np.array(intensity)

    top_indices = np.argsort(intensity_array)[-top_n:]

    return mz_array[top_indices].tolist()

def find_y1_b1_direct(mz, intensity, tolerance):
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
            # for nl_mass in [0, MASS_NH3, MASS_H2O]:
            for nl_mass in [0]:
                recovered_experimental_mz_mass = experimental_mz + nl_mass

                # singly charge in appraoch 1 because y1 and b1 have small mass containing only one AA. (more charge makes the mass more smaller)
                if abs(recovered_experimental_mz_mass - y1_expected) <=tolerance:
                    # removing the singly charge proton
                    y1_candidates[aa] = (experimental_mz - MASS_PROTON - MASS_H2O, experimental_mz, 1, nl_mass, False)

                if abs(recovered_experimental_mz_mass - b1_expected) <=tolerance:
                    # removing the singly charge proton
                    b1_candidates[aa] = (experimental_mz - MASS_PROTON, experimental_mz, 1, nl_mass, False)

    return y1_candidates, b1_candidates

def find_y1_b1_back_calc(mz, intensity, precursor_mz, precursor_charge, tolerance):
    y1_candidates = {}
    b1_candidates = {}

    precursor_neutral_mass = (precursor_mz * precursor_charge) - (precursor_charge* MASS_PROTON)
    # make single charged
    precursor_neutral_mass_singly = precursor_neutral_mass + MASS_PROTON
    # print(f"\nprecursor_neutral_mass_singly: {precursor_neutral_mass_singly}")
    # print(f"precursor_charge: {precursor_charge}")
    
    if intensity is None:
        filter_mz = mz
    else:
        filter_mz = filter_top_peaks(mz, intensity,150)

    # Filtering and keeping only the peaks that is greater than the midpoint of the precursor mass. We only want y(n-1) and b(n-1) here
    mz_midpoint = precursor_mz * 0.5
    # print(f"mz_midpoint: {mz_midpoint}")
    large_mz = [m for m in filter_mz if m >= mz_midpoint]

    # mz_array = np.array(mz)
    # threshold = mz_array.max() * 0.2
    # high_mass_peaks = mz_array[mz_array >=threshold]

    for aa, mass in AA_MASSES.items():
        # y1 and b1 ions are singly charged
        y1_expected = mass + MASS_H2O
        b1_expected = mass
        for large_experimental_mz in large_mz:
            for charge in range(1, precursor_charge+1):

                large_experimental_mz_neutral = large_experimental_mz * charge - (charge * MASS_PROTON)  
                for nl_mass in [0, MASS_NH3, MASS_H2O]:
                    recovered_large_experimental_mz_mass = large_experimental_mz_neutral + nl_mass
                    #make singly charged
                    recovered_large_experimental_mz_mass_singly = recovered_large_experimental_mz_mass + MASS_PROTON
                 
                    derived_AA = precursor_neutral_mass_singly - recovered_large_experimental_mz_mass_singly

                    # derived_AA does not exist in the input raw mz, so we have to create it as charge = 1. 
                    y1_expected_singly_charged = y1_expected + MASS_PROTON
                    b1_expected_singly_charged = b1_expected + MASS_PROTON

                    if abs(derived_AA - y1_expected) <=tolerance:
                        y1_candidates[aa] = (derived_AA - MASS_H2O, y1_expected_singly_charged, charge, nl_mass, True)

                    if abs(derived_AA - b1_expected) <=tolerance:
                        b1_candidates[aa] = (derived_AA, b1_expected_singly_charged, charge, nl_mass, True)

    return y1_candidates, b1_candidates

# def spectrum_to_input_text(mz_list, intensity_list):
#     pairs = sorted(zip(mz_list, intensity_list))
#     lines = ["Experimental MS/MS spectrum (m/z, intesity):"]
#     lines += [f"m/z {mz:5f} intensity {intensity}" for mz, intensity in pairs]
#     lines.append("Analyze this spectrum and identify the most likely peptide sequence.")
#     return "\n".join(lines)

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

        mz_list = scan_annot['experimental_mass'].tolist()
        # mz_list = scan_annot['theoretical_mass'].tolist()

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

def process_prospect(annotation_data_path, meta_data_path, correct_instances_path, tolerance, termination_tolerance, num_examples, save_dataset):
    spectra = load_prospect(annotation_data_path, meta_data_path, correct_instances_path, num_examples)

    total = 0
    y_correct = 0
    b_correct = 0
    both_correct = 0
    y_or_b_correct = 0
    # correct_instances = []
    dataset = []

    def normalize(s):
        return s.replace('I', 'L')

    for spec in spectra:
        total += 1
        peptide = spec['peptide']
        true_y = normalize(peptide)
        true_b = normalize(peptide)


        y_seqs, b_seqs, reasoning_trace = derive_peptide(
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

        if y_match and b_match: both_correct += 1
        if y_match or b_match: y_or_b_correct+= 1
        # if y_match or b_match:
        #     correct_instances.append({
        #         'scan_number': spec['scan_number'],
        #         'raw_file':spec['raw_file'],
        #         'peptide':peptide
        #     })

        if save_dataset:
            dataset.append({
            'input': spectrum_to_input_text(spec['mz'], spec['precursor_mz'], spec['precursor_charge']),
            'output': reasoning_trace,
            'peptide': peptide,
            'scan_number': spec['scan_number'],
            'raw_file': spec['raw_file'],
            'y_b_match': y_match or b_match
            })
        
            # Store correct peptides
            if total % 1000 == 0:
                with open('output/reasoning_dataset_2000.json', 'w') as f:
                    json.dump(dataset, f, indent=2)
                print(f"Progress: {total} processed, {y_or_b_correct} correct so for")

    print("Results:")
    print(f"Total: {total}")
    print(f"Y correct: {y_correct} ({100*y_correct/total:.1f}%)")
    print(f"B correct: {b_correct} ({100*b_correct/total:.1f}%)")
    print(f"Both correct: {both_correct} ({100*both_correct/total:.1f}%)")
    print(f"Y or B correct: {y_or_b_correct} ({100*y_or_b_correct/total:.1f}%)")

    if save_dataset:
        print(f"Dataset size: {len(dataset)}")

        with open('output/reasoning_dataset_2000.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved {len(dataset)} training examples to output/reasoning_dataset_2000.json")
        # # Store correct peptides
        # pd.DataFrame(correct_instances).to_csv('output/correct_instances.csv', index=False)
        # print(f"Saved {len(correct_instances)} correct instances")
              


if __name__ == "__main__":

    isProspect = True
    filterCorrectPeptides = False

    if isProspect:
        if filterCorrectPeptides:
            correct_instances_path = "/lclhome/apali007/projects/spectra_algorithm/output/correct_instances_TUM_third_pool_1_01_01_annotation.csv"
        else:
            correct_instances_path = ""
        annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
        meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
        OUTPUT_PATH = ""
        TOLERANCE = 0.005
        TERMINATION_TOLERANCE= 0.005
        VERBOSE = True
        NUM_EXAMPLES = 100
        SAVE_DATASET = False
        # NUM_EXAMPLES = 7317

        process_prospect(annotation_data_path, meta_data_path, correct_instances_path, tolerance=TOLERANCE, termination_tolerance=TERMINATION_TOLERANCE, num_examples=NUM_EXAMPLES, save_dataset = SAVE_DATASET)

    else:
        MGF_PATH = "/disk/raptor-array/SaeedLab-Data/Mass-Spec/SpectralLibraries/Massive-KB-v1/LIBRARY_CREATION_AUGMENT_LIBRARY_TEST-82c0124b-download_filtered_mgf_library-main.mgf"
        OUTPUT_PATH = "output/bottom_up_spectra_training_data.json"
        TOLERANCE = 0.02
        TERMINATION_TOLERANCE= 0.01
        VERBOSE = False
        NUM_EXAMPLES = None

        process_mgf(MGF_PATH, OUTPUT_PATH, tolerance=TOLERANCE, termination_tolerance=TERMINATION_TOLERANCE, num_examples=NUM_EXAMPLES)