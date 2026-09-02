# constant values from https://github.com/wilhelm-lab/spectrum_fundamentals/blob/development/spectrum_fundamentals/constants.py 
AA_MASSES = {

    "A": 71.037114,
    "R": 156.101111,
    "N": 114.042927,
    "D": 115.026943,

    "C": 103.009185,
    "E": 129.042593,
    "Q": 128.058578,
    "G": 57.021464,

    "H": 137.058912,
    "I": 113.084064,
    "L": 113.084064,
    "K": 128.094963,

    "M": 131.040485,
    "F": 147.068414,
    "P": 97.052764,
    "S": 87.032028,

    "T": 101.047679,
    "W": 186.079313,
    "Y": 163.063329,
    "V": 99.068414,

}

NEUTRAL_LOSS_MASSES = {
    'H2O': 18.01056467,
    'NH3': 17.026549105
}

MASS_PROTON =  1.007276467

TOLERANCE_BY_ANALYZER = {
    'FTMS' : 0.04,
    'ITMS' : 0.5,
}

DEFAULT_PEAK_TOLERANCE =  0.2

PRECURSOR_TOLERANCE = 0.1

# Case labels for derive_peptide outcomes
CASE_CLEAN_B_ION_LADDER = "CASE_CLEAN_B_ION_LADDER"
CASE_CLEAN_Y_ION_LADDER = "CASE_CLEAN_Y_ION_LADDER"
CASE_COMPLEMENTARY_GAPS = "CASE_COMPLEMENTARY_GAPS"
CASE_UNRESOLVABLE_MULTI_GAP = "CASE_UNRESOLVABLE_MULTI_GAP"
CASE_SINGLE_GAP_RESOLVED = "CASE_SINGLE_GAP_RESOLVED"
CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH = "CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH"


CASES = [
    CASE_CLEAN_B_ION_LADDER, CASE_CLEAN_Y_ION_LADDER, CASE_COMPLEMENTARY_GAPS, CASE_UNRESOLVABLE_MULTI_GAP, 
    CASE_UNRESOLVABLE_SINGLE_GAP_MASS_MISMATCH, CASE_SINGLE_GAP_RESOLVED    
]

AA_MASS_TABLE_TEXT = "\n".join(
    f"{aa}: {mass:.5f} Da" for aa, mass in AA_MASSES.items()
)

SYSTEM_PROMPT = (
    "You are an expert in de novo peptide sequencing from tandem mass spectrometry (MS/MS) "
    "proteomics data. Given a spectrum's precursor mass, precursor charge, mass analyzer and a list of fragment "
    "ion peaks (experimental mass and relative intensity), determine the most likely peptide sequence.\n\n"
    "Use the following procedure:\n"
    "1. First, attempt to derive the sequence using the y-ion ladder (C-terminus to N-terminus). "
    "For each position, identify the fragment peak, verify its mass against the theoretical fragment mass "
    f"within a tolerance that depends on the spectrum's mass analyzer: {TOLERANCE_BY_ANALYZER['FTMS']} Da for FTMS and "
    f"{TOLERANCE_BY_ANALYZER['ITMS']} Da for ITMS. Then compute the amino acid residue from "
    "the mass difference between consecutive fragments. If the y-ion ladder resolves the complete "
    "sequence, validate it against the precursor mass "
    "and stop. No need to derive a b-ion ladder if the sequence is already confirmed.\n"
    "2. When a terminal residue (the first or last position) has no fragment peak, recover it from "
        "the adjacent known fragment and the precursor mass rather than reporting it as missing.\n"
    "3. If gaps remain in the y-ion ladder, attempt the b-ion ladder (N-terminus to C-terminus) the "
    "same way. If the b-ion ladder alone resolves the complete sequence, validate it against the "
    "precursor mass and stop.\n"
    "4. If both ladders have gaps, merge them: for each unresolved b-ion position, check whether the "
    "y-ion ladder resolved that same position. If merging resolves the complete "
    "sequence, validate it against the precursor mass.\n"
    "5. If exactly one residue remains unresolved after merging, recover it by subtracting the sum "
    "of the known residues and one water molecule from the precursor mass.\n"
    "6. If two or more residues remain unresolved after merging, state clearly that the complete "
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

# Thse sample input are from /lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/input_data/test_split.json
SAMPLE_INPUT = "Precursor mz: 667.85479 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z, intensity):" \
    "\nm/z 147.11246, intensity 0.31000\nm/z 185.05533, intensity 0.10000\nm/z 195.03891, intensity 0.03000\nm/z 212.06584, intensity 0.14000" \
    "\nm/z 213.05009, intensity 0.12000\nm/z 218.14929, intensity 0.26000\nm/z 230.07646, intensity 1.00000\nm/z 278.11417, intensity 0.02000" \
    "\nm/z 330.10748, intensity 0.02000\nm/z 331.23291, intensity 0.10000\nm/z 348.11713, intensity 0.13000\nm/z 364.72824, intensity 0.01000" \
    "\nm/z 375.13028, intensity 0.02000\nm/z 376.11121, intensity 0.08000\nm/z 393.13873, intensity 0.15000\nm/z 405.76367, intensity 0.09000" \
    "\nm/z 406.26431, intensity 0.02000\nm/z 414.27802, intensity 0.11000\nm/z 419.15717, intensity 0.03000\nm/z 448.76730, intensity 0.02000" \
    "\nm/z 470.30945, intensity 0.02000\nm/z 471.78757, intensity 0.08000\nm/z 473.13153, intensity 0.04000\nm/z 485.22443, intensity 0.03000" \
    "\nm/z 487.33368, intensity 0.49000\nm/z 490.15421, intensity 0.04000\nm/z 491.13788, intensity 0.09000\nm/z 493.73444, intensity 0.02000" \
    "\nm/z 494.22699, intensity 0.11000\nm/z 494.73117, intensity 0.02000\nm/z 502.74188, intensity 0.24000\nm/z 508.16580, intensity 0.18000" \
    "\nm/z 544.31299, intensity 0.02000\nm/z 544.80511, intensity 0.05000\nm/z 545.29236, intensity 0.06000\nm/z 550.77295, intensity 0.02000" \
    "\nm/z 553.32050, intensity 0.44000\nm/z 553.82349, intensity 0.14000\nm/z 558.31573, intensity 0.02000\nm/z 559.28375, intensity 0.04000" \
    "\nm/z 569.37494, intensity 0.03000\nm/z 586.29449, intensity 0.03000\nm/z 586.40192, intensity 0.47000\nm/z 601.17957, intensity 0.02000" \
    "\nm/z 601.82806, intensity 0.06000\nm/z 610.34058, intensity 0.40000\nm/z 618.21326, intensity 0.02000\nm/z 619.19519, intensity 0.05000" \
    "\nm/z 636.22620, intensity 0.12000\nm/z 682.45514, intensity 0.04000\nm/z 687.25146, intensity 0.02000\nm/z 699.48553, intensity 0.43000" \
    "\nm/z 704.29138, intensity 0.02000\nm/z 715.25244, intensity 0.03000\nm/z 731.29834, intensity 0.02000\nm/z 732.28296, intensity 0.06000" \
    "\nm/z 749.30914, intensity 0.12000\nm/z 796.46094, intensity 0.05000\nm/z 810.51990, intensity 0.16000\nm/z 811.52142, intensity 0.04000" \
    "\nm/z 827.54382, intensity 0.55000\nm/z 831.35925, intensity 0.03000\nm/z 848.36932, intensity 0.03000\nm/z 924.56531, intensity 0.05000" \
    "\nm/z 925.55377, intensity 0.08000\nm/z 942.57092, intensity 0.82000\nm/z 959.51770, intensity 0.05000\nm/z 976.49286, intensity 0.01000" \
    "\nm/z 986.47412, intensity 0.02000\nm/z 987.46826, intensity 0.03000\nm/z 1004.47638, intensity 0.24000\nm/z 1073.55994, intensity 0.02000" \
    "\nm/z 1087.62561, intensity 0.03000\nm/z 1088.60034, intensity 0.03000\nm/z 1089.56531, intensity 0.11000\nm/z 1090.55603, intensity 0.02000" \
    "\nm/z 1100.52588, intensity 0.03000\nm/z 1105.63049, intensity 0.29000\nm/z 1106.63342, intensity 0.08000\nm/z 1117.57019, intensity 0.16000" \
    "\nm/z 1118.56238, intensity 0.04000\nm/z 1160.59924, intensity 0.06000\nm/z 1171.57361, intensity 0.03000\nm/z 1188.59583, intensity 0.32000" \
    "\nAnalyze this spectrum and identify the most likely peptide sequence."