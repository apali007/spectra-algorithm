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
SAMPLE_INPUT = "Precursor mz: 590.27440 Da, Precursor Charge: 3, Mass Analyzer: FTMS\nExperimental MS/MS spectrum (m/z, intensity):\nm/z 147.07545, intensity 0.02000\nm/z 147.11298, intensity 0.28000\nm/z 157.06073, intensity 0.12000\nm/z 167.04440, intensity 0.02000\nm/z 181.06052, intensity 0.19000\nm/z 185.05542, intensity 0.17000\nm/z 186.08713, intensity 0.02000\nm/z 195.07538, intensity 0.03000\nm/z 199.07127, intensity 0.27000\nm/z 209.05574, intensity 0.03000\nm/z 216.09767, intensity 0.76000\nm/z 226.08195, intensity 0.45000\nm/z 227.06531, intensity 0.10000\nm/z 230.11470, intensity 0.03000\nm/z 244.09254, intensity 0.63000\nm/z 244.12440, intensity 0.02000\nm/z 252.10918, intensity 0.03000\nm/z 262.13977, intensity 0.25000\nm/z 268.09229, intensity 0.05000\nm/z 292.64197, intensity 0.07000\nm/z 296.08716, intensity 0.07000\nm/z 310.10361, intensity 0.03000\nm/z 314.09833, intensity 0.07000\nm/z 323.14371, intensity 0.02000\nm/z 326.14426, intensity 0.03000\nm/z 328.11230, intensity 0.02000\nm/z 333.17636, intensity 0.16000\nm/z 336.16040, intensity 0.10000\nm/z 337.11301, intensity 0.13000\nm/z 338.09848, intensity 0.06000\nm/z 346.15881, intensity 0.02000\nm/z 355.12314, intensity 0.09000\nm/z 356.10757, intensity 0.10000\nm/z 364.66461, intensity 0.02000\nm/z 372.15128, intensity 0.14000\nm/z 373.13519, intensity 0.25000\nm/z 382.13428, intensity 0.02000\nm/z 384.18961, intensity 0.02000\nm/z 392.70334, intensity 0.09000\nm/z 397.13544, intensity 0.12000\nm/z 430.19281, intensity 0.06000\nm/z 438.17007, intensity 0.10000\nm/z 440.21628, intensity 0.04000\nm/z 447.21948, intensity 0.44000\nm/z 448.21719, intensity 0.21000\nm/z 448.71747, intensity 0.04000\nm/z 450.19907, intensity 0.07000\nm/z 456.17783, intensity 0.03000\nm/z 457.22250, intensity 0.12000\nm/z 460.56076, intensity 0.03000\nm/z 465.17368, intensity 0.05000\nm/z 466.15985, intensity 0.06000\nm/z 466.23801, intensity 0.03000\nm/z 467.14285, intensity 0.02000\nm/z 468.21692, intensity 0.07000\nm/z 483.18082, intensity 0.09000\nm/z 484.16849, intensity 0.04000\nm/z 495.22647, intensity 0.02000\nm/z 496.22110, intensity 0.02000\nm/z 501.19501, intensity 0.08000\nm/z 503.24191, intensity 0.07000\nm/z 503.73505, intensity 0.07000\nm/z 504.23331, intensity 0.03000\nm/z 511.17688, intensity 0.19000\nm/z 512.24628, intensity 0.21000\nm/z 512.73938, intensity 0.42000\nm/z 513.23859, intensity 0.14000\nm/z 521.25073, intensity 0.22000\nm/z 523.26184, intensity 0.04000\nm/z 524.25189, intensity 0.02000\nm/z 525.20508, intensity 0.11000\nm/z 529.19379, intensity 0.04000\nm/z 532.27161, intensity 0.02000\nm/z 552.25372, intensity 0.02000\nm/z 560.27411, intensity 0.08000\nm/z 561.75928, intensity 0.02000\nm/z 566.26733, intensity 0.05000\nm/z 567.24713, intensity 0.08000\nm/z 568.26892, intensity 0.03000\nm/z 568.78455, intensity 0.03000\nm/z 569.27783, intensity 0.04000\nm/z 577.25769, intensity 0.02000\nm/z 577.79272, intensity 0.36000\nm/z 578.27209, intensity 0.15000\nm/z 580.20679, intensity 0.02000\nm/z 581.18286, intensity 0.05000\nm/z 582.78558, intensity 0.06000\nm/z 583.28400, intensity 0.03000\nm/z 584.27832, intensity 0.48000\nm/z 589.80249, intensity 0.14000\nm/z 590.28992, intensity 0.26000\nm/z 590.78949, intensity 0.10000\nm/z 595.28296, intensity 0.06000\nm/z 596.26959, intensity 0.15000\nm/z 598.21704, intensity 0.05000\nm/z 613.30023, intensity 0.02000\nm/z 615.23779, intensity 0.14000\nm/z 616.78302, intensity 0.05000\nm/z 617.29187, intensity 0.05000\nm/z 617.78790, intensity 0.06000\nm/z 624.25739, intensity 0.10000\nm/z 625.26465, intensity 0.05000\nm/z 625.80286, intensity 0.02000\nm/z 626.30005, intensity 0.14000\nm/z 634.81494, intensity 0.33000\nm/z 636.27441, intensity 0.06000\nm/z 638.28589, intensity 0.05000\nm/z 639.29132, intensity 0.03000\nm/z 640.29944, intensity 0.04000\nm/z 653.29749, intensity 0.08000\nm/z 654.28510, intensity 0.06000\nm/z 656.29810, intensity 0.02000\nm/z 671.31030, intensity 1.00000\nm/z 672.81342, intensity 0.05000\nm/z 673.30542, intensity 0.04000\nm/z 681.32104, intensity 0.09000\nm/z 681.81879, intensity 0.20000\nm/z 682.32050, intensity 0.08000\nm/z 687.30841, intensity 0.02000\nm/z 689.83875, intensity 0.29000\nm/z 690.33307, intensity 0.63000\nm/z 692.30090, intensity 0.06000\nm/z 693.28522, intensity 0.03000\nm/z 694.27460, intensity 0.03000\nm/z 698.84473, intensity 0.17000\nm/z 703.83478, intensity 0.04000\nm/z 704.32526, intensity 0.08000\nm/z 704.83221, intensity 0.03000\nm/z 707.30890, intensity 0.04000\nm/z 710.32648, intensity 0.02000\nm/z 711.29901, intensity 0.04000\nm/z 724.31989, intensity 0.07000\nm/z 725.31726, intensity 0.04000\nm/z 728.31897, intensity 0.09000\nm/z 729.31934, intensity 0.02000\nm/z 736.33783, intensity 0.02000\nm/z 736.83820, intensity 0.03000\nm/z 737.33215, intensity 0.06000\nm/z 745.35461, intensity 0.05000\nm/z 745.84650, intensity 0.15000\nm/z 746.34558, intensity 0.11000\nm/z 752.33551, intensity 0.02000\nm/z 754.36005, intensity 0.30000\nm/z 754.85931, intensity 0.26000\nm/z 760.33850, intensity 0.02000\nm/z 763.36218, intensity 0.33000\nm/z 766.38037, intensity 0.05000\nm/z 767.34076, intensity 0.10000\nm/z 768.34344, intensity 0.08000\nm/z 768.85535, intensity 0.03000\nm/z 780.34961, intensity 0.02000\nm/z 784.39459, intensity 0.64000\nm/z 794.37579, intensity 0.03000\nm/z 809.88806, intensity 0.03000\nm/z 810.37054, intensity 0.05000\nm/z 810.86963, intensity 0.02000\nm/z 818.88745, intensity 0.06000\nm/z 819.38605, intensity 0.07000\nm/z 824.36371, intensity 0.12000\nm/z 827.89014, intensity 0.07000\nm/z 856.38593, intensity 0.02000\nm/z 877.40033, intensity 0.05000\nm/z 881.36066, intensity 0.04000\nm/z 895.42407, intensity 0.28000\nm/z 896.41040, intensity 0.13000\nm/z 913.43665, intensity 0.88000\nm/z 923.41418, intensity 0.05000\nm/z 992.46112, intensity 0.02000\nm/z 1005.46033, intensity 0.05000\nm/z 1006.46423, intensity 0.10000\nm/z 1007.45740, intensity 0.09000\nm/z 1008.47302, intensity 0.05000\nm/z 1009.44574, intensity 0.02000\nm/z 1023.48395, intensity 0.13000\nm/z 1024.47156, intensity 0.46000\nm/z 1041.49463, intensity 0.53000\nm/z 1051.47632, intensity 0.09000\nm/z 1119.53918, intensity 0.02000\nm/z 1122.51624, intensity 0.03000\nm/z 1137.55774, intensity 0.03000\nm/z 1154.58264, intensity 0.23000\nm/z 1164.54749, intensity 0.03000\nm/z 1165.54822, intensity 0.02000\nm/z 1233.56128, intensity 0.04000\nm/z 1234.56458, intensity 0.04000\nm/z 1250.58801, intensity 0.04000\nm/z 1251.58801, intensity 0.06000\nm/z 1268.62207, intensity 0.19000\nm/z 1279.59277, intensity 0.02000\nm/z 1475.66687, intensity 0.02000\nAnalyze this spectrum and identify the most likely peptide sequence."