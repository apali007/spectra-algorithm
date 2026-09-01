import pickle, pandas as pd

AA = {
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
H2O, PROTON = 18.01056467, 1.007276467

spectra = pickle.load(open("/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_spectra_prospect_cache.pkl", "rb"))

rows = []
for spec in spectra:
    pep = spec['peptide']
    mz, charge = spec['precursor_mz'], spec['precursor_charge']

    exp_neutral = mz * charge - charge * PROTON
    theo_neutral = sum(AA[a] for a in pep) + H2O
    gap = abs(exp_neutral - theo_neutral)

    rows.append({
        'gap': gap,
        'scan_number': spec['scan_number'],
        'raw_file': spec['raw_file'],
        'modified_sequence': pep,
    })

df = pd.DataFrame(rows).sort_values('gap', ascending=False)

meta = pd.read_parquet("/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet")[
    ['scan_number', 'raw_file', 'modified_sequence', 'mass_analyzer', 'fragmentation']
]
df = df.merge(meta, on=['scan_number', 'raw_file', 'modified_sequence'], how='left')

print(f"rows before merge: {len(rows)}, after merge: {len(df)}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.head(50).to_string(index=False))
print("\nPrecursor worst case gap summary by mass analyzer:")
print(df.groupby('mass_analyzer')['gap'].describe(percentiles=[0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 0.999]))