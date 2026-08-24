import pickle, pandas as pd

AA = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}
H2O, PROTON = 18.01056, 1.00728

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
print("\nPrecursor gap summary by mass analyzer:")
print(df.groupby('mass_analyzer')['gap'].describe())