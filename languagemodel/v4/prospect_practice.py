# # ================
# # Task: Spectra table from PROSPECT to CSV

# import pandas as pd

# annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
# annot_data = pd.read_parquet(annotation_data_path)

# rows = annot_data[
#     (annot_data['peptide_sequence'] == 'PQGPPPQEGNK') & 
#     (annot_data['scan_number'] == 4999) & 
#     (annot_data['raw_file'] == '01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1')
#     ]

# rows.to_csv("output/PQGPPPQEGNK.csv", index=False)

# # ================


# ================
# Task: Print a table

import pandas as pd

meta_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool_meta_data.parquet"
meta_data = pd.read_parquet(meta_data_path)

# annotation_data_path = "/lclhome/apali007/projects/spectra_algorithm/prospect/data/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet"
# annot_data = pd.read_parquet(annotation_data_path)

# print(meta_data.head(20))

# print(meta_data.columns.tolist())

dupes = meta_data.groupby(['scan_number', 'raw_file']).size()
dupe_keys = dupes[dupes > 1].index
sample = meta_data.set_index(['scan_number', 'raw_file']).loc[dupe_keys[:5]]
print(sample[['modified_sequence', 'andromeda_score']])

# ================
