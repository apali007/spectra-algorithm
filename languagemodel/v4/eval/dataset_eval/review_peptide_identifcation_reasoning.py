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

precursor_mass = 590.27440
precursor_charge = 3
# peptide_sequence = "ICPLNLEISHNADK"
# peptide_sequence = "NEEQNLQELSHNADK"
peptide_sequence = "ICPBNLQEISHNADK"

precursor_neutral_mass = precursor_mass * precursor_charge - precursor_charge * MASS_PROTON
aa_residues = list(peptide_sequence)

print(f"Precursor neutral mass: {precursor_neutral_mass}")

aa_sum = sum(AA_MASSES[aa] for aa in peptide_sequence if aa in AA_MASSES)

aa_sum_water = aa_sum + NEUTRAL_LOSS_MASSES['H2O']

print(f"aa_sum plus water: {aa_sum_water}")

print(f"diff: {precursor_neutral_mass - aa_sum_water}.")

# Predicted_aa = precursor_neutral_mass - prev_mz + MASS_PROTON - NEUTRAL_LOSS_MASSES['H2O']

# print(f"Predicted aa: {Predicted_aa}")

# print(f"diff: {Predicted_aa - AA_MASSES[aa_residues[len(aa_residues)-1]]}")