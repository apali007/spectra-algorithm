# ============
# Count the input length.

from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

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

MASS_PROTON =  1.00728
TOLERANCE = 0.2
PRECURSOR_TOLERANCE = 0.1

AA_MASS_TABLE_TEXT = "\n".join(
    f"{aa}: {mass:.5f} Da" for aa, mass in AA_MASSES.items()
)
SYSTEM_PROMPT = (
    "You are an expert in de novo peptide sequencing from tandem mass spectrometry (MS/MS) "
    "proteomics data. Given a spectrum's precursor mass, precursor charge, and a list of fragment "
    "ion peaks (experimental mass and relative intensity), determine the most likely peptide sequence.\n\n"
    "Use the following procedure:\n"
    "1. First, attempt to derive the sequence using the y-ion ladder (C-terminus to N-terminus). "
    "For each position, identify the fragment peak, verify its mass against the theoretical fragment mass "
    f"within a tolerance of {TOLERANCE} Da, and compute the amino acid residue from "
    "the mass difference between consecutive fragments. If the y-ion ladder resolves the complete "
    "sequence, validate it against the precursor mass "
    "and stop. No need to derive a b-ion ladder if the sequence is already confirmed.\n"
    "2. If gaps remain in the y-ion ladder, attempt the b-ion ladder (N-terminus to C-terminus) the "
    "same way. If the b-ion ladder alone resolves the complete sequence, validate it against the "
    "precursor mass and stop.\n"
    "3. If both ladders have gaps, merge them: for each unresolved b-ion position, check whether the "
    "y-ion ladder resolved that same position. If merging resolves the complete "
    "sequence, validate it against the precursor mass.\n"
    "4. If exactly one residue remains unresolved after merging, recover it by subtracting the sum "
    "of the known residues and one water molecule from the precursor mass. Check whether the "
    f"resulting mass matches any residue mass within the fragment tolerance ({TOLERANCE} Da). If it does, "
    "predict the full sequence and validate it against the precursor mass.\n"
    "5. When a terminal residue (the first or last position) has no fragment peak, recover it from "
    "the adjacent known fragment and the precursor mass rather than reporting it as missing.\n"
    "6. If two or more residues remain unresolved after merging, or the mass recovered in step 4 "
    f"does not match any amino acid residue within tolerance of {TOLERANCE} Da, state clearly that the complete "
    "sequence cannot be determined, report the partial sequence with unresolved positions marked, "
    "and show the arithmetic that led to that conclusion.\n\n"
    "Amino acid residue masses (monoisotopic, Da):\n"
    f"{AA_MASS_TABLE_TEXT}\n\n"
    f"Proton mass: {MASS_PROTON} Da. Water (H2O) mass: {NEUTRAL_LOSS_MASSES['H2O']} Da. "
    f"Ammonia (NH3) mass: {NEUTRAL_LOSS_MASSES['NH3']} Da.\n\n"
    "When a fragment ion has charge greater than 1, convert it to a singly-charged mass before using "
    "it in any calculation: singly_charged_mass = (m/z x charge) - (charge - 1) x proton_mass.\n\n"
    "When a fragment ion involves a neutral loss, add the corresponding neutral loss mass (H2O or NH3) "
    "to the singly-charged mass: mass_after_correction = singly_charged_mass + neutral_loss_mass.\n\n"
    f"Always validate any complete candidate sequence against the precursor mass "
    f"within a precursor tolerance of {PRECURSOR_TOLERANCE} Da before reporting it "
    "as final. Show your arithmetic explicitly at every step."
)

def get_train_real_full_length_stats(path, label):
    with open(path) as f:
        data = json.load(f)

    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": entry['input']},
            {"role": "assistant", "content": entry['output']},
        ]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - REAL chat-template FULL (system+input+output) length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")
    over_limit = sum(1 for l in lengths if l > 12288)
    print(f" examples exceeding 12288: {over_limit} ({100*over_limit/n:.2f}%)")

def get_test_real_input_length_stats(path, label):
    with open(path) as f:
        data = json.load(f)
    
    lengths = []
    for entry in data:
        messages = [
            {"role": "system", "content":SYSTEM_PROMPT},
            {"role": "user", "content": entry['input']},
        ]

        tokens = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True)
        lengths.append(len(tokens))

    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    print(f"\n{label} (n={n}) - REAL chat-template input length:")
    print(f" min: {lengths_sorted[0]}")
    print(f" mean: {sum(lengths)/n:.0f}")
    print(f" median: {lengths_sorted[n//2]}")
    print(f" 95th pct: {lengths_sorted[int(n*0.95)]}")
    print(f" 99th pct: {lengths_sorted[int(n*0.99)]}")
    print(f" max: {lengths_sorted[-1]}")

get_train_real_full_length_stats(
    '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/30k_all_cases_prospect_reasoning_dataset_train.json',
    "Train set"
)

get_test_real_input_length_stats(
    '/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/30k_all_cases_prospect_reasoning_dataset_test.json',
    "Test set"
)

