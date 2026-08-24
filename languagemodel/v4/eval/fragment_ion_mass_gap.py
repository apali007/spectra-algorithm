import pickle
import pandas as pd

spectra = pickle.load(open("/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_spectra_prospect_cache.pkl", "rb"))

rows = []
for spec in spectra:
    a = spec['annotation_rows']
    gap = (a['experimental_mass'] - a['theoretical_mass']).abs()
    i = gap.idxmax()

    rows.append({
        'gap': gap[i],
        'ion': f"{a.loc[i, 'ion_type']}{a.loc[i, 'no']}",
        'scan_number': spec['scan_number'],
        'raw_file': spec['raw_file'],
        'modified_sequence': spec['peptide']
    })

df = pd.DataFrame(rows).sort_values('gap', ascending = False)

meta = pd.read_parquet("/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet")[
    ['scan_number', 'raw_file', 'modified_sequence', 'mass_analyzer', 'fragmentation']
    ]

df = df.merge(meta, on=['scan_number', 'raw_file', 'modified_sequence'], how='left')

print(df.head(50).to_string(index=False))

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("\nGap summary by mass analyzer:")
print(df.groupby('mass_analyzer')['gap'].describe())

print(f"\nrows before merge: {len(rows)}, after merge: {len(df)}")